import type { FeatureMeta, Inputs } from "./types";

export type FormValues = Record<string, string>;

/**
 * Converts raw form strings into an API `inputs` object, driven entirely by the backend
 * feature metadata (no field lists or encodings here). Empty optional fields are omitted;
 * the backend sends them to the model as missing.
 */
export function buildInputs(
  features: FeatureMeta[],
  values: FormValues,
): { inputs: Inputs; errors: Record<string, string> } {
  const inputs: Inputs = {};
  const errors: Record<string, string> = {};

  for (const f of features) {
    const raw = (values[f.name] ?? "").trim();
    if (raw === "") {
      if (f.required) errors[f.name] = "Required";
      continue;
    }
    if (f.allowed_values) {
      if (!f.allowed_values.includes(raw)) errors[f.name] = "Choose one of the options";
      else inputs[f.name] = raw;
      continue;
    }
    const n = Number(raw);
    if (!Number.isFinite(n)) {
      errors[f.name] = "Enter a number";
      continue;
    }
    const { min, max } = f.hard_bounds;
    if (min !== null && n < min) errors[f.name] = `Must be at least ${min}`;
    else if (max !== null && n > max) errors[f.name] = `Must be at most ${max}`;
    else inputs[f.name] = n;
  }
  return { inputs, errors };
}

export function formatPercent(p: number): string {
  const pct = p * 100;
  if (pct > 0 && pct < 0.1) return "<0.1%";
  return `${pct.toFixed(pct >= 10 ? 1 : 2)}%`;
}

export function formatNumber(n: number): string {
  return Number.isInteger(n) ? String(n) : n.toFixed(2).replace(/\.?0+$/, "");
}
