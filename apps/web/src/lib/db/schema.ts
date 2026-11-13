import { jsonb, pgTable, text, timestamp, uuid, integer } from "drizzle-orm/pg-core";

export const restorationJobs = pgTable("restoration_jobs", {
  id: uuid("id").defaultRandom().primaryKey(),
  accessTokenHash: text("access_token_hash").notNull(),
  requesterHash: text("requester_hash"),
  status: text("status").notNull().default("queued"),
  sourcePathname: text("source_pathname").notNull(),
  resultPathname: text("result_pathname"),
  errorMessage: text("error_message"),
  options: jsonb("options").$type<{
    tileSize: number;
    tileOverlap: number;
    outputFormat: "png" | "jpeg";
    amp: boolean;
  }>().notNull().default({ tileSize: 0, tileOverlap: 32, outputFormat: "png", amp: false }),
  attempts: integer("attempts").notNull().default(0),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
  startedAt: timestamp("started_at", { withTimezone: true }),
  completedAt: timestamp("completed_at", { withTimezone: true }),
});
