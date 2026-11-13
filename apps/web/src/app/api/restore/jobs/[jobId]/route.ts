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
  if (!job) {
    return NextResponse.json({ error: "Job not found." }, { status: 404 });
  }

  return NextResponse.json(
    {
      jobId: job.id,
      status: job.status,
      attempts: job.attempts,
      error: job.errorMessage,
      resultAvailable: job.status === "succeeded" && Boolean(job.resultPathname),
    },
    { headers: { "Cache-Control": "no-store" } },
  );
}
