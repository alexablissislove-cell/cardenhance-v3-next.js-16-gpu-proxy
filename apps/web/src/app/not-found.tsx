import Link from "next/link";

export default function NotFound() {
  return (
    <main className="site-shell">
      <section className="intro" aria-labelledby="not-found-title">
        <div>
          <p className="micro-label">404 / ROUTE NOT FOUND</p>
          <h1 id="not-found-title">This surface isn&apos;t here.</h1>
          <p className="micro-label" style={{ marginTop: 24, letterSpacing: "normal" }}>
            The page may have moved, or the address may be mistyped.
          </p>
          <Link className="text-link" href="/" style={{ marginTop: 24 }}>
            Return to restoration studio
          </Link>
        </div>
      </section>
    </main>
  );
}
