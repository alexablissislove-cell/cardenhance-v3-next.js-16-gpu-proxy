import { createHash, timingSafeEqual } from "node:crypto";

import { del, get, put } from "@vercel/blob";
import { and, eq, isNull, lt, or, sql } from "drizzle-orm";

import { db } from "@/lib/db";
import { restorationJobs } from "@/lib/db/schema";
import { inferenceEndpoint, inferenceHeaders, maxImageBytes } from "@/lib/server/config";

const maxAttempts = 3;
const staleJobMs = 5 * 60 * 1000;

export function hashAccessToken(token: string): string {
  return createHash("sha256").update(token).digest("hex");
}

export async function authorizeJob(jobId: string, token: string) {
  if (!token || token.length > 128) return null;
  const [job] = await db
    .select()
    .from(restorationJobs)
    .where(eq(restorationJobs.id, jobId))
    .limit(1);

  if (!job) return null;
  const actual = Buffer.from(hashAccessToken(token), "hex");
  const expected = Buffer.from(job.accessTokenHash, "hex");
  if (actual.length !== expected.length || !timingSafeEqual(actual, expected)) return null;
  return job;
}

export async function processRestorationJob(jobId: string) {
  const endpoint = inferenceEndpoint("restore");
  if (!endpoint) throw new Error("The GPU restoration service is not configured.");

  const staleBefore = new Date(Date.now() - staleJobMs);
  const [job] = await db
    .update(restorationJobs)
    .set({
      status: "running",
      attempts: sql`${restorationJobs.attempts} + 1`,
      startedAt: new Date(),
      updatedAt: new Date(),
      errorMessage: null,
    })
    .where(
      and(
        eq(restorationJobs.id, jobId),
        lt(restorationJobs.attempts, maxAttempts),
        or(
          eq(restorationJobs.status, "queued"),
          eq(restorationJobs.status, "failed"),
          and(
            eq(restorationJobs.status, "running"),
            or(isNull(restorationJobs.startedAt), lt(restorationJobs.startedAt, staleBefore)),
          ),
        ),
      ),
    )
    .returning();

  if (!job) {
    const existing = await db
      .select({ status: restorationJobs.status, attempts: restorationJobs.attempts })
      .from(restorationJobs)
      .where(eq(restorationJobs.id, jobId))
      .limit(1);
    if (!existing[0]) throw new Error("Restoration job was not found.");
    return { status: existing[0].status, attempts: existing[0].attempts };
  }

  try {
    const source = await get(job.sourcePathname, { access: "private" });
    if (!source || source.statusCode !== 200) throw new Error("The source scan is unavailable.");

    const sourceBytes = Buffer.from(await new Response(source.stream).arrayBuffer());
    if (sourceBytes.length === 0 || sourceBytes.length > maxImageBytes()) {
      throw new Error("The source scan exceeds the configured size limit.");
    }

    const form = new FormData();
    form.append(
      "file",
      new File([sourceBytes], `${job.id}.image`, { type: source.blob.contentType }),
    );
    form.append("tile_size", String(job.options.tileSize));
    form.append("tile_overlap", String(job.options.tileOverlap));
    form.append("amp", String(job.options.amp));
    form.append("output_format", job.options.outputFormat);

    const upstream = await fetch(endpoint, {
      method: "POST",
      headers: inferenceHeaders(),
      body: form,
      cache: "no-store",
      signal: AbortSignal.timeout(290_000),
    });

    if (!upstream.ok) {
      let detail = `GPU service returned HTTP ${upstream.status}.`;
      try {
        const payload = (await upstream.json()) as { detail?: unknown };
        if (typeof payload.detail === "string") detail = payload.detail.slice(0, 300);
      } catch {
        // Keep the bounded HTTP status when the service response is not JSON.
      }
      throw new Error(detail);
    }

    const contentType = upstream.headers.get("content-type") ?? "";
    if (!contentType.startsWith("image/") || !upstream.body) {
      throw new Error("GPU service returned an invalid image.");
    }

    const resultPathname = `restoration-jobs/${job.id}/result.${job.options.outputFormat}`;
    const resultBlob = await put(resultPathname, upstream.body, {
      access: "private",
      addRandomSuffix: false,
      contentType,
    });
    const inferenceMsValue = Number(upstream.headers.get("x-inference-ms"));
    const inferenceMs = Number.isFinite(inferenceMsValue) && inferenceMsValue > 0
      ? Math.round(inferenceMsValue)
      : null;

    await db
      .update(restorationJobs)
      .set({
        status: "succeeded",
        resultPathname: resultBlob.pathname,
        errorMessage: null,
        completedAt: new Date(),
        updatedAt: new Date(),
      })
      .where(eq(restorationJobs.id, job.id));
    await del(job.sourcePathname).catch(() => undefined);

    return { status: "succeeded", attempts: job.attempts + 1, inferenceMs };
  } catch (error) {
    const message = error instanceof Error ? error.message.slice(0, 500) : "Restoration failed.";
    await db
      .update(restorationJobs)
      .set({ status: "failed", errorMessage: message, updatedAt: new Date() })
      .where(eq(restorationJobs.id, job.id));

    if (job.attempts + 1 >= maxAttempts) {
      await del(job.sourcePathname).catch(() => undefined);
    }
    throw error;
  }
}
