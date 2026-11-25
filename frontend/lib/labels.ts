// Display-only labels. They never change the values sent to the API: the form still submits the
// backend's own option strings / codes.

const OPTION_LABELS: Record<string, Record<string, string>> = {
  // NHANES education codes (DMDEDUC2); the backend accepts the numeric code 1-5.
  education_level: {
    "1": "Less than 9th grade",
    "2": "9–11th grade",
    "3": "High school / GED",
    "4": "Some college / AA",
    "5": "College graduate or above",
  },
};

export function capitalize(s: string): string {
  return s.length ? s[0].toUpperCase() + s.slice(1) : s;
}

/** Fields rendered as a labelled select instead of a free number input. */
export function hasOptionLabels(field: string): boolean {
  return field in OPTION_LABELS;
}

export function optionLabels(field: string): [string, string][] {
  return Object.entries(OPTION_LABELS[field] ?? {});
}

export function optionLabel(field: string, value: string | number): string {
  return OPTION_LABELS[field]?.[String(value)] ?? capitalize(String(value));
}

/** Human-readable rendering of a submitted value (enums capitalised, education decoded, units appended). */
export function displayValue(field: string, value: string | number, unit?: string | null): string {
  if (field in OPTION_LABELS) return optionLabel(field, value);
  if (typeof value === "string") return capitalize(value);
  const text = Number.isInteger(value) ? String(value) : String(Number(value.toFixed(2)));
  return unit ? `${text} ${unit}` : text;
}