import { defineWorkflowTool } from "eve/tools";
import { z } from "zod";

import { processRestorationJob } from "../../src/lib/server/restoration-jobs";

export default defineWorkflowTool({
  description:
    "Process one CardEnhance restoration job by its exact job ID. The job was created by the app and its source image is stored privately. Call this tool only when the user explicitly asks you to process that job. Never substitute a different ID or expose the image access token.",
  inputSchema: z.object({ jobId: z.string().uuid() }),
  async task({ jobId }) {
    "use workflow";
    return await processJobStep(jobId);
  },
});

async function processJobStep(jobId: string) {
  "use step";
  return processRestorationJob(jobId);
}
