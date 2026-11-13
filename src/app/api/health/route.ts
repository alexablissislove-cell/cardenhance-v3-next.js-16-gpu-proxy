import { inferenceEndpoint, inferenceHeaders, maxImageBytes } from "@/lib/server/config";
import type { ServiceStatus } from "@/lib/types";

export const dynamic = "force-dynamic";

export async function GET(): Promise<Response> {
  const endpoint = inferenceEndpoint("health");
  const base: ServiceStatus = {
    configured: endpoint !== null,
    ready: false,
    maxBytes: maxImageBytes(),
  };
  if (!endpoint) {
    return Response.json(base, { headers: { "Cache-Control": "no-store" } });
  }

  try {
    const response = await fetch(endpoint, {
      headers: inferenceHeaders(),
      cache: "no-store",
      signal: AbortSignal.timeout(5_000),
    });
    if (!response.ok) {
      return Response.json(
        { ...base, detail: `Inference service returned HTTP ${response.status}` },
        { headers: { "Cache-Control": "no-store" } },
      );
    }
    const health = await response.json() as Record<string, unknown>;
    return Response.json(
      {
        ...base,
        ready: health.status === "ready",
        mode: health.mode,
        generator: health.generator,
        device: health.device,
        checkpointEpoch: health.checkpoint_epoch,
        detail: health.status === "ready" ? undefined : "Checkpoint is not ready",
      },
      { headers: { "Cache-Control": "no-store" } },
    );
  } catch {
    return Response.json(
      { ...base, detail: "Inference service is unreachable" },
      { headers: { "Cache-Control": "no-store" } },
    );
  }
}
