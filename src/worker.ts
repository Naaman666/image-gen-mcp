import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { WebStandardStreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js";
import { z } from "zod";

interface Env {
  AI: {
    run(model: string, inputs: Record<string, unknown>): Promise<{ image: string }>;
  };
  IMAGES: KVNamespace;
  API_SECRET?: string;
}

function randomId(): string {
  return crypto.randomUUID().replace(/-/g, "");
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    if (url.pathname === "/health") {
      return Response.json({ ok: true, service: "image-gen-mcp" });
    }

    // Serve stored images: GET /img/:id
    const imgMatch = url.pathname.match(/^\/img\/([a-f0-9]+)$/);
    if (imgMatch && request.method === "GET") {
      const data = await env.IMAGES.get(imgMatch[1], "arrayBuffer");
      if (!data) return new Response("Not Found", { status: 404 });
      return new Response(data, {
        headers: { "Content-Type": "image/jpeg", "Cache-Control": "public, max-age=3600" },
      });
    }

    if (url.pathname === "/mcp" || url.pathname.startsWith("/mcp/")) {
      return handleMcp(request, env);
    }

    return new Response("Not Found", { status: 404 });
  },
};

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
  const server = new McpServer({ name: "image-gen", version: "1.1.0" });

  server.tool(
    "generate_image",
    "Generate an image from a text prompt using Cloudflare Workers AI (Flux 1 Schnell). Returns a JPEG image URL you can open in a browser.",
    {
      prompt: z.string().min(1).max(2048).describe("Text description of the image to generate"),
      steps: z
        .number()
        .int()
        .min(1)
        .max(8)
        .optional()
        .default(4)
        .describe("Diffusion steps (1–8). Higher = better quality but slower. Default: 4"),
    },
    async ({ prompt, steps }) => {
      const result = await env.AI.run("@cf/black-forest-labs/flux-1-schnell", { prompt, steps });

      // Store image in KV with 1-hour TTL
      const id = randomId();
      const imageBytes = base64ToBytes(result.image);
      await env.IMAGES.put(id, imageBytes, { expirationTtl: 3600 });

      const imageUrl = `${host}/img/${id}`;

      return {
        content: [
          {
            type: "text" as const,
            text: `Image generated! Open in browser: ${imageUrl}\n\nPrompt: "${prompt}"`,
          },
        ],
      };
    },
  );

  const transport = new WebStandardStreamableHTTPServerTransport();
  await server.connect(transport);
  return transport.handleRequest(request);
}

function base64ToBytes(base64: string): Uint8Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}
