import { afterEach, describe, expect, it, vi } from "vitest";

import { GET } from "@/app/api/health/route";

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe("GET /api/health", () => {
  it("returns an unconfigured status without an upstream request", async () => {
    vi.stubEnv("RESTORATION_API_URL", "");
    vi.stubEnv("MAX_IMAGE_BYTES", "4000000");
    const fetchMock = vi.fn<typeof fetch>();
    vi.stubGlobal("fetch", fetchMock);

    const response = await GET();

    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({
      configured: false,
      ready: false,
      maxBytes: 4000000,
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("maps checkpoint metadata and authenticates upstream", async () => {
    vi.stubEnv("RESTORATION_API_URL", "https://gpu.example.test");
    vi.stubEnv("RESTORATION_API_KEY", "health-secret");
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      Response.json({
        status: "ready",
        mode: "paired",
        generator: "rrdb",
        device: "cuda:0",
        checkpoint_epoch: 42,
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const response = await GET();

    expect(await response.json()).toMatchObject({
      configured: true,
      ready: true,
      mode: "paired",
      generator: "rrdb",
      device: "cuda:0",
      checkpointEpoch: 42,
    });
    expect(String(fetchMock.mock.calls[0][0])).toBe("https://gpu.example.test/health");
    expect(fetchMock.mock.calls[0][1]?.headers).toEqual({ Authorization: "Bearer health-secret" });
    expect(fetchMock.mock.calls[0][1]?.cache).toBe("no-store");
  });

  it("turns an unavailable upstream into a visible service status", async () => {
    vi.stubEnv("RESTORATION_API_URL", "https://gpu.example.test");
    vi.stubGlobal("fetch", vi.fn<typeof fetch>().mockRejectedValue(new Error("offline")));

    const response = await GET();

    expect(response.status).toBe(200);
    expect(await response.json()).toMatchObject({
      configured: true,
      ready: false,
      detail: "Inference service is unreachable",
    });
  });
});
