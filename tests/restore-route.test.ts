import { afterEach, describe, expect, it, vi } from "vitest";

import { POST } from "@/app/api/restore/route";

function uploadRequest(fields: Record<string, string | File> = {}): Request {
  const body = new FormData();
  body.append(
    "image",
    fields.image ?? new File([Uint8Array.from([1, 2, 3])], "damaged.png", { type: "image/png" }),
  );
  for (const [key, value] of Object.entries(fields)) {
    if (key !== "image") body.append(key, value);
  }
  return new Request("http://localhost/api/restore", { method: "POST", body });
}

function configureService(): void {
  vi.stubEnv("RESTORATION_API_URL", "https://gpu.example.test/");
  vi.stubEnv("RESTORATION_API_KEY", "server-secret");
  vi.stubEnv("MAX_IMAGE_BYTES", "4000000");
}

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe("POST /api/restore", () => {
  it("reports missing inference configuration without calling upstream", async () => {
    vi.stubEnv("RESTORATION_API_URL", "");
    const fetchMock = vi.fn<typeof fetch>();
    vi.stubGlobal("fetch", fetchMock);

    const response = await POST(uploadRequest());

    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({
      error: "Set RESTORATION_API_URL to your deployed GPU inference service.",
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("rejects unsupported image types before forwarding", async () => {
    configureService();
    const fetchMock = vi.fn<typeof fetch>();
    vi.stubGlobal("fetch", fetchMock);
    const request = uploadRequest({
      image: new File(["not an image"], "notes.txt", { type: "text/plain" }),
    });

    const response = await POST(request);

    expect(response.status).toBe(415);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("rejects uploads above the configured byte limit", async () => {
    configureService();
    vi.stubEnv("MAX_IMAGE_BYTES", "10");
    const fetchMock = vi.fn<typeof fetch>();
    vi.stubGlobal("fetch", fetchMock);
    const request = uploadRequest({
      image: new File([new Uint8Array(11)], "large.png", { type: "image/png" }),
    });

    const response = await POST(request);

    expect(response.status).toBe(413);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("forwards validated settings and only the server-side credential", async () => {
    configureService();
    const restoredBytes = Uint8Array.from([137, 80, 78, 71]);
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(restoredBytes, {
        status: 200,
        headers: {
          "Content-Type": "image/png",
          "X-Checkpoint-Mode": "paired",
          "X-Generator": "rrdb",
          "X-Inference-Ms": "1250",
          "Set-Cookie": "upstream=must-not-leak",
        },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const request = uploadRequest({
      tileSize: "512",
      tileOverlap: "64",
      outputFormat: "png",
      amp: "true",
    });
    request.headers.set("Cookie", "browser=must-not-forward");
    request.headers.set("Authorization", "Bearer browser-token");

    const response = await POST(request);

    expect(response.status).toBe(200);
    expect(response.headers.get("Content-Type")).toBe("image/png");
    expect(response.headers.get("X-Inference-Ms")).toBe("1250");
    expect(response.headers.get("Set-Cookie")).toBeNull();
    expect(new Uint8Array(await response.arrayBuffer())).toEqual(restoredBytes);

    expect(fetchMock).toHaveBeenCalledOnce();
    const [input, init] = fetchMock.mock.calls[0];
    expect(String(input)).toBe("https://gpu.example.test/restore");
    expect(init?.method).toBe("POST");
    expect(init?.headers).toEqual({ Authorization: "Bearer server-secret" });
    expect(init?.signal).toBe(request.signal);
    const forwarded = init?.body as FormData;
    expect(forwarded.get("tile_size")).toBe("512");
    expect(forwarded.get("tile_overlap")).toBe("64");
    expect(forwarded.get("output_format")).toBe("png");
    expect(forwarded.get("amp")).toBe("true");
    const image = forwarded.get("file");
    expect(image).toBeInstanceOf(File);
    expect((image as File).name).toBe("damaged.png");
  });

  it("omits authorization when no server credential is configured", async () => {
    configureService();
    vi.stubEnv("RESTORATION_API_KEY", "");
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(new Uint8Array([1]), { headers: { "Content-Type": "image/png" } }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const response = await POST(uploadRequest());

    expect(response.status).toBe(200);
    expect(fetchMock.mock.calls[0][1]?.headers).toEqual({});
  });

  it("passes through bounded upstream client errors", async () => {
    configureService();
    vi.stubGlobal("fetch", vi.fn<typeof fetch>().mockResolvedValue(
      Response.json({ detail: "No checkpoint is loaded" }, { status: 409 }),
    ));

    const response = await POST(uploadRequest());

    expect(response.status).toBe(409);
    expect(await response.json()).toEqual({ error: "No checkpoint is loaded" });
  });

  it("maps upstream server failures to a bad-gateway response", async () => {
    configureService();
    vi.stubGlobal("fetch", vi.fn<typeof fetch>().mockResolvedValue(
      new Response("GPU failed", { status: 500 }),
    ));

    const response = await POST(uploadRequest());

    expect(response.status).toBe(502);
    expect(await response.json()).toEqual({ error: "Inference service returned HTTP 500" });
  });

  it("rejects a non-image success response from upstream", async () => {
    configureService();
    vi.stubGlobal("fetch", vi.fn<typeof fetch>().mockResolvedValue(
      Response.json({ ok: true }),
    ));

    const response = await POST(uploadRequest());

    expect(response.status).toBe(502);
    expect(await response.json()).toEqual({ error: "Inference service did not return an image." });
  });
});
