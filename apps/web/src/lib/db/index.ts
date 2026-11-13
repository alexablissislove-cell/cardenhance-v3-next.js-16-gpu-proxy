import { Pool } from "pg";
import { drizzle } from "drizzle-orm/node-postgres";

import * as schema from "./schema";

const globalForDatabase = globalThis as typeof globalThis & {
  restorationPool?: Pool;
};

export const pool =
  globalForDatabase.restorationPool ??
  new Pool({ connectionString: process.env.DATABASE_URL, max: 5 });

if (process.env.NODE_ENV !== "production") {
  globalForDatabase.restorationPool = pool;
}

export const db = drizzle(pool, { schema });
