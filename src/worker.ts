import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { WebStandardStreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js";
import { z } from "zod";

interface Env {
  AI: {
    run(model: string, inputs: Record<string, unknown>): Promise<unknown>;
  };
  IMAGES: KVNamespace;
  API_SECRET?: string;
  HUGGINGFACE_API_KEY?: string;
}

interface ImageMeta {
  mimeType: string;
}

// ── Model definitions ────────────────────────────────────────────────────────

const CF_MODELS = {
  "cf-flux-schnell": {
    id: "@cf/black-forest-labs/flux-1-schnell",
    stepsParam: "steps" as const,
    defaultSteps: 8,
    maxSteps: 8,
  },
  "cf-sdxl": {
    id: "@cf/stabilityai/stable-diffusion-xl-base-1.0",
    stepsParam: "num_steps" as const,
    defaultSteps: 20,
    maxSteps: 20,
  },
  "cf-dreamshaper": {
    id: "@cf/lykon/dreamshaper-8-lcm",
    stepsParam: "num_steps" as const,
    defaultSteps: 8,
    maxSteps: 20,
  },
} as const;

// ── Helpers ──────────────────────────────────────────────────────────────────

function detectMimeType(bytes: Uint8Array): string {
  if (bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47) {
    return "image/png";
  }
  return "image/jpeg";
}

function base64ToBytes(b64: string): Uint8Array {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

async function streamToBytes(stream: ReadableStream<Uint8Array>): Promise<Uint8Array> {
  const reader = stream.getReader();
  const chunks: Uint8Array[] = [];
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) chunks.push(value);
  }
  const total = chunks.reduce((n, c) => n + c.length, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) { out.set(chunk, offset); offset += chunk.length; }
  return out;
}

async function anyToBytes(result: unknown): Promise<Uint8Array> {
  // Flux Schnell path — { image: string } base64 JSON
  if (result && typeof (result as Record<string, unknown>).image === "string") {
    return base64ToBytes((result as { image: string }).image);
  }
  // ReadableStream path (SDXL, DreamShaper)
  if (result instanceof ReadableStream) return streamToBytes(result);
  if (result instanceof ArrayBuffer) return new Uint8Array(result);
  if (result instanceof Response) return new Uint8Array(await result.arrayBuffer());
  // Last resort
  return streamToBytes(result as ReadableStream<Uint8Array>);
}

// ── Image generation ─────────────────────────────────────────────────────────

async function generateCf(
  env: Env,
  modelKey: keyof typeof CF_MODELS,
  prompt: string,
  steps?: number,
): Promise<Uint8Array> {
  const model = CF_MODELS[modelKey];
  const result = await env.AI.run(model.id, {
    prompt,
    [model.stepsParam]: steps ?? model.defaultSteps,
  });
  return anyToBytes(result);
}

async function generateHf(env: Env, prompt: string): Promise<Uint8Array> {
  if (!env.HUGGINGFACE_API_KEY) throw new Error("HUGGINGFACE_API_KEY not configured");
  const resp = await fetch("https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.HUGGINGFACE_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ inputs: prompt }),
  });
  if (!resp.ok) throw new Error(`HuggingFace API error: ${resp.status} ${await resp.text()}`);
  return new Uint8Array(await resp.arrayBuffer());
}

// ── Worker entry point ───────────────────────────────────────────────────────

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    if (url.pathname === "/health") {
      return Response.json({ ok: true, service: "image-gen-mcp", version: "2.0.0" });
    }

    const imgMatch = url.pathname.match(/^\/img\/([a-f0-9-]+)$/);
    if (imgMatch && request.method === "GET") {
      const { value: data, metadata } = await env.IMAGES.getWithMetadata<ImageMeta>(imgMatch[1], "arrayBuffer");
      if (!data) return new Response("Not Found", { status: 404 });
      return new Response(data, {
        headers: {
          "Content-Type": metadata?.mimeType ?? "image/jpeg",
          "Cache-Control": "public, max-age=3600",
        },
      });
    }

    if (url.pathname === "/mcp" || url.pathname.startsWith("/mcp/")) {
      return handleMcp(request, env);
    }

    return new Response("Not Found", { status: 404 });
  },
};

// ── MCP handler ──────────────────────────────────────────────────────────────

async function handleMcp(request: Request, env: Env): Promise<Response> {
  if (request.method === "OPTIONS") {
    return new Response(null, {
      status: 204,
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization, Mcp-Session-Id",
      },
    });
  }

  if (env.API_SECRET) {
    const auth = request.headers.get("authorization") ?? "";
    const token = auth.startsWith("Bearer ") ? auth.slice(7).trim() : "";
    if (token !== env.API_SECRET) {
      return new Response(JSON.stringify({ error: "unauthorized" }), {
        status: 401,
        headers: { "Content-Type": "application/json" },
      });
    }
  }

  const host = new URL(request.url).origin;
  const server = new McpServer({ name: "image-gen", version: "2.0.0" });

  server.tool(
    "generate_image",
    [
      "Generate an image from a text prompt.",
      "Models: cf-flux-schnell (fast JPEG, default), cf-sdxl (Stable Diffusion XL, PNG),",
      "cf-dreamshaper (artistic/portrait, PNG), hf-flux-dev (highest quality FLUX.1-dev via HuggingFace, JPEG).",
      "Returns a browser-openable URL valid for 1 hour.",
    ].join(" "),
    {
      prompt: z.string().min(1).max(2048).describe("Text description of the image"),
      model: z
        .enum(["cf-flux-schnell", "cf-sdxl", "cf-dreamshaper", "hf-flux-dev"])
        .optional()
        .default("cf-flux-schnell")
        .describe("Which model to use. Default: cf-flux-schnell"),
      steps: z
        .number()
        .int()
        .min(1)
        .max(20)
        .optional()
        .describe("Diffusion steps. Model defaults: cf-flux-schnell=8, cf-sdxl=20, cf-dreamshaper=8. Ignored for hf-flux-dev."),
    },
    async ({ prompt, model, steps }) => {
      let imageBytes: Uint8Array;

      if (model === "hf-flux-dev") {
        imageBytes = await generateHf(env, prompt);
      } else {
        imageBytes = await generateCf(env, model, prompt, steps);
      }

      const mimeType = detectMimeType(imageBytes);
      const id = crypto.randomUUID().replace(/-/g, "");
      await env.IMAGES.put(id, imageBytes, {
        expirationTtl: 3600,
        metadata: { mimeType } satisfies ImageMeta,
      });

      const modelLabel: Record<string, string> = {
        "cf-flux-schnell": "Cloudflare Flux Schnell",
        "cf-sdxl": "Cloudflare SDXL",
        "cf-dreamshaper": "Cloudflare DreamShaper",
        "hf-flux-dev": "HuggingFace FLUX.1-dev",
      };

      return {
        content: [
          {
            type: "text" as const,
            text: [
              `Image generated with ${modelLabel[model]}!`,
              `Open in browser: ${host}/img/${id}`,
              ``,
              `Prompt: "${prompt}"`,
            ].join("\n"),
          },
        ],
      };
    },
  );

  const transport = new WebStandardStreamableHTTPServerTransport();
  await server.connect(transport);
  return transport.handleRequest(request);
}
