"use client";

import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { AlertCircle, ChevronDown, FlaskConical, Info, RotateCcw, Sparkles } from "lucide-react";
import type { FeatureMeta, Mode } from "@/lib/types";
import { buildInputs, formatNumber, type FormValues } from "@/lib/form";
import { hasOptionLabels, optionLabel, optionLabels } from "@/lib/labels";
import ModeToggle from "./ModeToggle";
import { FLOW, SPRING, TAP } from "@/lib/motion";

export interface SampleProfile {
  id: string;
  label: string;
  values: FormValues;
}

// Presentation grouping only. Fields the backend returns that are not listed here fall into the
// laboratory group, so new backend fields still render without a code change.
const GROUPS: { id: string; title: string; names: string[] }[] = [
  { id: "profile", title: "Basic profile", names: ["age", "gender", "bmi", "waist_circumference"] },
  { id: "vitals", title: "Vital signs", names: ["mean_sbp", "mean_dbp", "mean_pulse"] },
  {
    id: "lifestyle",
    title: "Lifestyle & background",
    names: [
      "smoking_status",
      "alcohol_frequency",
      "physical_activity_level",
      "sedentary_minutes",
      "education_level",
      "poverty_income_ratio",
    ],
  },
];

function Field({
  meta,
  value,
  error,
  onChange,
}: {
  meta: FeatureMeta;
  value: string;
  error?: string;
  onChange: (v: string) => void;
}) {
  const id = `f-${meta.name}`;
  const { min, max } = meta.hard_bounds;
  const wide = meta.allowed_values !== null && meta.allowed_values.length > 2;

  const label = (
    <label className="field-label" htmlFor={id}>
      {meta.label}
      {meta.required && (
        <span className="req" title="Required" aria-label="required">
          *
        </span>
      )}
      {meta.description && (
        <span className="info" title={meta.description}>
          <Info size={13} />
        </span>
      )}
    </label>
  );

  if (meta.allowed_values) {
    return (
      <div className={`field${wide ? " wide" : ""}`}>
        {label}
        <div className="chips" role="radiogroup" aria-label={meta.label} id={id}>
          {meta.allowed_values.map((opt) => (
            <motion.button
              key={opt}
              whileTap={TAP}
              type="button"
              role="radio"
              aria-checked={value === opt}
              className={`chip${error ? " invalid" : ""}`}
              // Optional choices can be cleared by selecting the active option again.
              onClick={() => onChange(value === opt && !meta.required ? "" : opt)}
            >
              {optionLabel(meta.name, opt)}
            </motion.button>
          ))}
        </div>
        {error && <span className="field-error">{error}</span>}
      </div>
    );
  }

  if (hasOptionLabels(meta.name)) {
    return (
      <div className="field wide">
        {label}
        <div className="input-wrap">
          <select
            id={id}
            className={`input select${error ? " invalid" : ""}${value === "" ? " placeholder" : ""}`}
            value={value}
            aria-invalid={!!error}
            onChange={(e) => onChange(e.target.value)}
          >
            <option value="">{meta.required ? "Select…" : "Not specified"}</option>
            {optionLabels(meta.name).map(([code, text]) => (
              <option key={code} value={code}>
                {text}
              </option>
            ))}
          </select>
        </div>
        {error && <span className="field-error">{error}</span>}
      </div>
    );
  }

  const range = meta.training_range
    ? `typical ${formatNumber(meta.training_range[0])}–${formatNumber(meta.training_range[1])}`
    : min !== null && max !== null
      ? `${formatNumber(min)}–${formatNumber(max)}`
      : null;

  return (
    <div className="field">
      {label}
      <div className="input-wrap">
        <input
          id={id}
          className={`input${error ? " invalid" : ""}`}
          type="number"
          inputMode="decimal"
          step="any"
          min={min ?? undefined}
          max={max ?? undefined}
          value={value}
          placeholder={meta.required ? "Required" : "Optional"}
          aria-invalid={!!error}
          onChange={(e) => onChange(e.target.value)}
        />
        {meta.unit && <span className="input-unit">{meta.unit}</span>}
      </div>
      {error ? <span className="field-error">{error}</span> : range && <span className="field-hint">{range}</span>}
    </div>
  );
}

export default function AssessmentForm({
  mode,
  onModeChange,
  features,
  featuresError,
  featuresLoading,
  onRetry,
  values,
  onValueChange,
  onFill,
  onReset,
  profiles,
  saveToHistory,
  onSaveToHistoryChange,
  submitting,
  onSubmit,
  serverErrors,
}: {
  mode: Mode;
  onModeChange: (m: Mode) => void;
  features: FeatureMeta[] | null;
  featuresError: string | null;
  featuresLoading: boolean;
  onRetry: () => void;
  values: FormValues;
  onValueChange: (name: string, value: string) => void;
  onFill: (p: SampleProfile) => void;
  onReset: () => void;
  profiles: SampleProfile[];
  saveToHistory: boolean;
  onSaveToHistoryChange: (v: boolean) => void;
  submitting: boolean;
  onSubmit: (values: FormValues) => void;
  serverErrors: string[];
}) {
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [labsOpen, setLabsOpen] = useState(false);
  const [labsTouched, setLabsTouched] = useState(false);

  const grouped = useMemo(() => {
    const known = new Set(GROUPS.flatMap((g) => g.names));
    const byName = new Map((features ?? []).map((f) => [f.name, f]));
    const main = GROUPS.map((g) => ({
      ...g,
      fields: g.names.map((n) => byName.get(n)).filter((f): f is FeatureMeta => !!f),
    })).filter((g) => g.fields.length > 0);
    const labs = (features ?? []).filter((f) => !known.has(f.name));
    return { main, labs };
  }, [features]);

  const labsFilled = grouped.labs.filter((f) => (values[f.name] ?? "").trim() !== "").length;

  // Open the lab section automatically when a sample profile fills it (unless the user toggled it).
  useEffect(() => {
    if (!labsTouched && labsFilled > 0) setLabsOpen(true);
  }, [labsFilled, labsTouched]);

  // A mode switch drops stale validation messages.
  useEffect(() => {
    setErrors({});
  }, [mode]);

  function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!features) return;
    const { errors: found } = buildInputs(features, values);
    setErrors(found);
    if (Object.keys(found).length > 0) {
      const firstLab = grouped.labs.some((f) => found[f.name]);
      if (firstLab) setLabsOpen(true);
      return;
    }
    onSubmit(values);
  }

  const setValue = (name: string, v: string) => {
    if (errors[name]) setErrors((prev) => ({ ...prev, [name]: "" }));
    onValueChange(name, v);
  };

  return (
    <form className="glass card-pad" onSubmit={submit} noValidate>
      <div className="form-head">
        <ModeToggle mode={mode} onChange={onModeChange} />
        <div className="profile-row">
          <span className="label">Quick fill sample:</span>
          {profiles.map((p) => (
            <button key={p.id} type="button" className="btn-ghost" onClick={() => onFill(p)}>
              <Sparkles size={13} />
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {featuresLoading && !features && (
        <div className="stack" aria-busy>
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className="skeleton" style={{ height: 42 }} />
          ))}
        </div>
      )}

      {featuresError && (
        <div className="banner banner-error" role="alert">
          <AlertCircle size={16} />
          <div>
            <strong>Could not load the assessment form.</strong>
            <div>{featuresError}</div>
            <button type="button" className="btn-ghost" style={{ marginTop: 8 }} onClick={onRetry}>
              <RotateCcw size={13} /> Try again
            </button>
          </div>
        </div>
      )}

      {features && (
        <>
          {grouped.main.map((g) => (
            <section className="field-group" key={g.id}>
              <div className="group-title">{g.title}</div>
              <div className="fields">
                {g.fields.map((f) => (
                  <Field
                    key={f.name}
                    meta={f}
                    value={values[f.name] ?? ""}
                    error={errors[f.name]}
                    onChange={(v) => setValue(f.name, v)}
                  />
                ))}
              </div>
            </section>
          ))}

          {grouped.labs.length > 0 && (
            <section className="field-group">
              <button
                type="button"
                className="collapse-head"
                aria-expanded={labsOpen}
                onClick={() => {
                  setLabsTouched(true);
                  setLabsOpen((o) => !o);
                }}
              >
                <span className="collapse-title">
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                    <FlaskConical size={16} /> Laboratory biomarkers
                  </span>
                  <small>Optional — add any results you have</small>
                </span>
                <span className="collapse-right">
                  {labsFilled > 0 && <span className="count-badge">{labsFilled} added</span>}
                  <motion.span animate={{ rotate: labsOpen ? 180 : 0 }} style={{ display: "inline-flex" }}>
                    <ChevronDown size={18} />
                  </motion.span>
                </span>
              </button>
              <AnimatePresence initial={false}>
                {labsOpen && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={FLOW}
                    style={{ overflow: "hidden" }}
                  >
                    <div className="fields" style={{ paddingTop: 14 }}>
                      {grouped.labs.map((f) => (
                        <Field
                          key={f.name}
                          meta={f}
                          value={values[f.name] ?? ""}
                          error={errors[f.name]}
                          onChange={(v) => setValue(f.name, v)}
                        />
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </section>
          )}

          {(Object.values(errors).some(Boolean) || serverErrors.length > 0) && (
            <div className="banner banner-error" role="alert" style={{ marginTop: 18 }}>
              <AlertCircle size={16} />
              <div>
                <strong>Please review the highlighted values.</strong>
                {serverErrors.length > 0 && (
                  <ul>
                    {serverErrors.map((s) => (
                      <li key={s}>{s}</li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          )}

          <div className="form-footer">
            <label className="check">
              <input type="checkbox" checked={saveToHistory} onChange={(e) => onSaveToHistoryChange(e.target.checked)} />
              Save this assessment to history
            </label>
            <div className="form-actions">
              <button type="button" className="btn-ghost" onClick={onReset} aria-label="Clear form">
                <RotateCcw size={14} /> Clear
              </button>
              <motion.button
                type="submit"
                className="btn-primary"
                disabled={submitting}
                whileHover={{ y: -2 }}
                whileTap={TAP}
                transition={SPRING}
              >
                {submitting ? (
                  <>
                    <span className="spinner" /> Calculating…
                  </>
                ) : (
                  "Estimate risk"
                )}
              </motion.button>
            </div>
          </div>
        </>
      )}
    </form>
  );
}
