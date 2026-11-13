import { createHmac, randomBytes, randomUUID } from "node:crypto";

import { del, put } from "@vercel/blob";
import { getVercelOidcToken } from "@vercel/oidc";
import { Client } from "eve/client";
import { NextResponse } from "next/server";
import { and, eq, gte, sql } from "drizzle-orm";

import { db } from "@/lib/db";
import { restorationJobs } from "@/lib/db/schema";
import { hashAccessToken } from "@/lib/server/restoration-jobs";
import { inferenceEndpoint, maxImageBytes } from "@/lib/server/config";
import { IMAGE_TYPES } from "@/lib/types";

export const runtime = "nodejs";
export const maxDuration = 60;

function errorResponse(error: string, status: number) {
  return NextResponse.json({ error }, { status, headers: { "Cache-Control": "no-store" } });
}

function readInteger(value: FormDataEntryValue | null, fallback: number): number | null {
  if (value === null) return fallback;
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) ? parsed : null;
}

function getRequesterHash(request: Request): string {
  const forwardedFor = request.headers.get("x-forwarded-for");
  const requester =
    request.headers.get("x-real-ip")?.trim() ||
    forwardedFor?.split(",").at(-1)?.trim() ||
    "unknown";
  return createHmac("sha256", process.env.DATABASE_URL ?? "cardenhance-local")
    .update(requester)
    .digest("hex");
}

export async function POST(request: Request) {
  if (!inferenceEndpoint("restore")) {
    return errorResponse("Set RESTORATION_API_URL to your deployed GPU inference service.", 503);
  }

  let form: FormData;
  try {
    form = await request.formData();
  } catch {
    return errorResponse("The upload must be valid multipart form data.", 400);
  }

  const image = form.get("image");
  if (!(image instanceof File)) return errorResponse("Choose an image to restore.", 400);
  if (!IMAGE_TYPES.includes(image.type as (typeof IMAGE_TYPES)[number])) {
    return errorResponse("Only JPEG, PNG, and WebP images are supported.", 415);
  }
  if (image.size === 0 || image.size > maxImageBytes()) {
    return errorResponse(
      `Image must be between 1 byte and ${Math.floor(maxImageBytes() / 1_000_000)} MB.`,
      413,
    );
  }

  const tileSize = readInteger(form.get("tileSize"), 0);
  const tileOverlap = readInteger(form.get("tileOverlap"), 32);
  const outputFormat = form.get("outputFormat") ?? "png";
  const amp = form.get("amp") === "true";
  if (tileSize === null || ![0, 256, 512, 1024].includes(tileSize)) {
    return errorResponse("Invalid tile size.", 400);
  }
  if (
    tileOverlap === null ||
    tileOverlap < 0 ||
    tileOverlap > 256 ||
    (tileSize > 0 && tileOverlap >= tileSize)
  ) {
    return errorResponse("Tile overlap must be non-negative and smaller than the tile.", 400);
  }
  if (outputFormat !== "png" && outputFormat !== "jpeg") {
    return errorResponse("Output format must be PNG or JPEG.", 400);
  }

  const id = randomUUID();
  const accessToken = randomBytes(32).toString("base64url");
  const extension = image.type === "image/jpeg" ? "jpg" : image.type === "image/webp" ? "webp" : "png";
  const sourcePathname = `restoration-jobs/${id}/source.${extension}`;

  try {
    const source = await put(sourcePathname, image, {
      access: "private",
      addRandomSuffix: false,
      contentType: image.type,
    });

    const requesterHash = getRequesterHash(request);
    const accepted = await db.transaction(async (transaction) => {
      await transaction.execute(
        sql`SELECT pg_advisory_xact_lock(hashtextextended(${requesterHash}, 0))`,
      );
      const [usage] = await transaction
        .select({ total: sql<number>`count(*)::int` })
        .from(restorationJobs)
        .where(
          and(
            eq(restorationJobs.requesterHash, requesterHash),
            gte(restorationJobs.createdAt, new Date(Date.now() - 60 * 60 * 1000)),
          ),
        );
      if ((usage?.total ?? 0) >= 20) return false;

      await transaction.insert(restorationJobs).values({
        id,
        accessTokenHash: hashAccessToken(accessToken),
        requesterHash,
        sourcePathname: source.pathname,
        options: { tileSize, tileOverlap, outputFormat, amp },
      });
      return true;
    });
    if (!accepted) {
      await del(source.pathname).catch(() => undefined);
      return errorResponse("Too many restoration jobs were submitted. Try again later.", 429);
    }

    const client = new Client({
      baseUrl: new URL(request.url).origin,
      ...(process.env.NODE_ENV === "production"
        ? { auth: { vercelOidc: { token: getVercelOidcToken } } }
        : {}),
    });
    const { session } = await client.sessions.create();
    await session.send(
      `Process the CardEnhance restoration job with ID ${id}. Use the process_restoration_job tool for this exact job ID, then report the resulting status.`,
    );

    return NextResponse.json(
      { jobId: id, accessToken, status: "queued" },
      { status: 202, headers: { "Cache-Control": "no-store" } },
    );
  } catch {
    await del(sourcePathname).catch(() => undefined);
    await db.delete(restorationJobs).where(eq(restorationJobs.id, id)).catch(() => undefined);
    return errorResponse("The restoration job could not be queued. Your source scan was removed.", 503);
  }
}
