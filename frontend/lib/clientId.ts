const STORAGE_KEY = "hra_client_id";

function generateId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  // Fallback for older browsers / non-secure contexts: random hex, not security-sensitive.
  const rnd = () => Math.floor(Math.random() * 0x100000000).toString(16).padStart(8, "0");
  return `${rnd()}${rnd()}${rnd()}${rnd()}`;
}

/**
 * Anonymous, browser-scoped identifier. It only separates one browser's saved history from another's;
 * it is not authentication. Returns "" during server rendering (no browser storage there).
 */
export function getClientId(): string {
  if (typeof window === "undefined") return "";
  try {
    let id = window.localStorage.getItem(STORAGE_KEY);
    if (!id) {
      id = generateId();
      window.localStorage.setItem(STORAGE_KEY, id);
    }
    return id;
  } catch {
    // Storage blocked (private mode): fall back to an id that lasts for this page session.
    return sessionFallback ?? (sessionFallback = generateId());
  }
}

let sessionFallback: string | null = null;