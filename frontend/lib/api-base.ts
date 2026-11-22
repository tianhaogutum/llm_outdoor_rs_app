/**
 * Base URL of the backend API.
 *
 * - NEXT_PUBLIC_API_URL wins when set at build time.
 * - Under `next dev` (page on port 3000/3001) the API runs separately on http://localhost:8000.
 * - Otherwise the page is the static export served by the backend itself, so the API is on the
 *   same origin and requests use relative URLs ("/api/...").
 */
export function resolveApiBaseUrl(): string {
  if (process.env.NEXT_PUBLIC_API_URL) return process.env.NEXT_PUBLIC_API_URL.replace(/\/$/, "");
  if (typeof window !== "undefined") {
    const port = window.location.port;
    if (port === "3000" || port === "3001") return "http://localhost:8000";
  }
  return "";
}

/** Human-readable backend location for error messages. */
export function describeApiBaseUrl(): string {
  const base = resolveApiBaseUrl();
  if (base) return base;
  return typeof window !== "undefined" ? window.location.origin : "the same origin";
}
