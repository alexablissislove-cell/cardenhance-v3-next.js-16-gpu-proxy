import { get } from "@vercel/blob";
import { NextResponse } from "next/server";

import { authorizeJob } from "@/lib/server/restoration-jobs";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function readToken(request: Request): string {
  const authorization = request.headers.get("authorization") ?? "";
  return authorization.startsWith("Bearer ") ? authorization.slice(7) : "";
}

export async function GET(
  request: Request,
  { params }: { params: Promise<{ jobId: string }> },
) {
  const { jobId } = await params;
  if (!/^[0-9a-f-]{36}$/i.test(jobId)) {
    return NextResponse.json({ error: "Job not found." }, { status: 404 });
  }

  const job = await authorizeJob(jobId, readToken(request));
  if (!job?.resultPathname || job.status !== "succeeded") {
    return NextResponse.json({ error: "Result not found." }, { status: 404 });
  }

  const result = await get(job.resultPathname, { access: "private" });
  if (!result || result.statusCode !== 200) {
    return NextResponse.json({ error: "Result not found." }, { status: 404 });
  }

  return new Response(result.stream, {
    headers: {
      "Content-Type": result.blob.contentType,
      "Cache-Control": "private, no-store",
      "Content-Disposition": `inline; filename="restored-${job.id}.${job.options.outputFormat === "jpeg" ? "jpg" : "png"}"`,
    },
  });
}
