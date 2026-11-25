"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { SPRING } from "@/lib/motion";
import { AlertCircle, CheckCircle2, ClipboardList, Info, TriangleAlert } from "lucide-react";
import Navbar, { type TabId } from "@/components/Navbar";
import AssessmentForm, { type SampleProfile } from "@/components/AssessmentForm";
import DiseaseResultCard, { DisclaimerNote } from "@/components/DiseaseResultCard";
import ResultsSkeleton from "@/components/ResultsSkeleton";
import HistoryView from "@/components/HistoryView";
import ModelPerformanceView from "@/components/ModelPerformanceView";
import { ApiError, getFeatures, predict } from "@/lib/api";
import { buildInputs, type FormValues } from "@/lib/form";
import type { FeatureMeta, Inputs, Mode, PredictResponse } from "@/lib/types";

/* Quick-fill demo profiles (sample data only; option strings are the backend's allowed values). */
const PROFILES: Record<Mode, SampleProfile[]> = {
  A: [
    {
      id: "a-lower",
      label: "Lower-risk profile",
      values: {
        age: "34", gender: "female", bmi: "22.4", waist_circumference: "74",
        mean_sbp: "108", mean_dbp: "68", mean_pulse: "66",
        smoking_status: "never", alcohol_frequency: "current", physical_activity_level: "high",
        sedentary_minutes: "240", education_level: "5", poverty_income_ratio: "4.1",
      },
    },
    {
      id: "a-higher",
      label: "Higher-risk profile",
      values: {
        age: "61", gender: "male", bmi: "33.1", waist_circumference: "113",
        mean_sbp: "146", mean_dbp: "92", mean_pulse: "82",
        smoking_status: "former", alcohol_frequency: "current", physical_activity_level: "low",
        sedentary_minutes: "540", education_level: "3", poverty_income_ratio: "1.6",
      },
    },
  ],
  B: [
    {
      id: "b-lower",
      label: "Lower-risk profile",
      values: {
        age: "38", gender: "female", bmi: "23.1", waist_circumference: "76",
        mean_sbp: "110", mean_dbp: "70", mean_pulse: "68",
        smoking_status: "never", alcohol_frequency: "current", physical_activity_level: "high",
        sedentary_minutes: "260", education_level: "5", poverty_income_ratio: "3.9",
        hba1c: "5.2", fasting_glucose: "88", total_cholesterol: "176", hdl_cholesterol: "63",
        triglycerides: "84", ldl_cholesterol: "98", serum_creatinine: "0.7", blood_urea_nitrogen: "12",
        serum_uric_acid: "4.2", alt_enzyme: "16", ast_enzyme: "18", hemoglobin: "13.4",
        wbc_count: "6.1", platelet_count: "240", rdw: "12.9", serum_albumin: "4.4",
      },
    },
    {
      id: "b-higher",
      label: "Higher-risk profile",
      values: {
        age: "63", gender: "male", bmi: "32.6", waist_circumference: "112",
        mean_sbp: "144", mean_dbp: "90", mean_pulse: "80",
        smoking_status: "former", alcohol_frequency: "current", physical_activity_level: "low",
        sedentary_minutes: "520", education_level: "3", poverty_income_ratio: "1.8",
        hba1c: "6.4", fasting_glucose: "122", total_cholesterol: "228", hdl_cholesterol: "38",
        triglycerides: "212", ldl_cholesterol: "148", serum_creatinine: "1.1", blood_urea_nitrogen: "19",
        serum_uric_acid: "7.0", alt_enzyme: "34", ast_enzyme: "29", hemoglobin: "14.6",
        wbc_count: "7.8", platelet_count: "262", rdw: "13.9", serum_albumin: "4.0",
      },
    },
  ],
};

export default function Home() {
  const [tab, setTab] = useState<TabId>("assessment");
  const [mode, setMode] = useState<Mode>("A");
  const [featuresByMode, setFeaturesByMode] = useState<Partial<Record<Mode, FeatureMeta[]>>>({});
  const [featuresLoading, setFeaturesLoading] = useState(false);
  const [featuresError, setFeaturesError] = useState<string | null>(null);
  const [values, setValues] = useState<FormValues>({});
  const [saveToHistory, setSaveToHistory] = useState(true);

  const [submitting, setSubmitting] = useState(false);
  const [serverErrors, setServerErrors] = useState<string[]>([]);
  const [error, setError] = useState<{ message: string; details: string[] } | null>(null);
  const [result, setResult] = useState<{ response: PredictResponse; submitted: Inputs } | null>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  const loadFeatures = useCallback(async (m: Mode) => {
    setFeaturesLoading(true);
    setFeaturesError(null);
    try {
      const data = await getFeatures(m);
      setFeaturesByMode((prev) => ({ ...prev, [m]: data.fields }));
    } catch (e) {
      setFeaturesError(e instanceof Error ? e.message : "Could not load the form.");
    } finally {
      setFeaturesLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!featuresByMode[mode]) void loadFeatures(mode);
  }, [mode, featuresByMode, loadFeatures]);

  const features = featuresByMode[mode] ?? null;

  const { labels, units } = useMemo(() => {
    const l: Record<string, string> = {};
    const u: Record<string, string | null> = {};
    for (const list of Object.values(featuresByMode)) {
      for (const f of list ?? []) {
        l[f.name] = f.label;
        u[f.name] = f.unit;
      }
    }
    return { labels: l, units: u };
  }, [featuresByMode]);

  async function handleSubmit(formValues: FormValues) {
    if (!features) return;
    const { inputs } = buildInputs(features, formValues);
    setSubmitting(true);
    setError(null);
    setServerErrors([]);
    try {
      const response = await predict({ mode, inputs, include_explanation: true, save_to_history: saveToHistory });
      setResult({ response, submitted: inputs });
      requestAnimationFrame(() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }));
    } catch (e) {
      if (e instanceof ApiError && e.status === 422) {
        setServerErrors(e.details);
      } else if (e instanceof ApiError) {
        setError({ message: e.message, details: e.details });
      } else {
        setError({ message: "Something went wrong.", details: [] });
      }
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="app-shell">
      <Navbar tab={tab} onChange={setTab} />
      <main className="container page">
        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={tab}
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={SPRING}
          >
        {tab === "assessment" && (
          <>
            <div className="page-head">
              <div className="eyebrow">Preventive screening support</div>
              <h1>Estimate risk for three conditions, independently.</h1>
              <p>
                Enter what you know. Each condition is assessed by its own model and reported with an estimated
                probability, the screening threshold it is compared against, and the factors that contributed most.
              </p>
            </div>

            <div className="workspace">
              <AssessmentForm
                mode={mode}
                onModeChange={(m) => {
                  setMode(m);
                  setServerErrors([]);
                }}
                features={features}
                featuresError={featuresError}
                featuresLoading={featuresLoading}
                onRetry={() => void loadFeatures(mode)}
                values={values}
                onValueChange={(name, v) => setValues((prev) => ({ ...prev, [name]: v }))}
                onFill={(p) => {
                  setValues(p.values);
                  setServerErrors([]);
                }}
                onReset={() => {
                  setValues({});
                  setServerErrors([]);
                }}
                profiles={PROFILES[mode]}
                saveToHistory={saveToHistory}
                onSaveToHistoryChange={setSaveToHistory}
                submitting={submitting}
                onSubmit={handleSubmit}
                serverErrors={serverErrors}
              />

              <section ref={resultsRef} style={{ scrollMarginTop: 90 }} aria-label="Results">
                {submitting ? (
                  <ResultsSkeleton />
                ) : error ? (
                  <div className="banner banner-error" role="alert">
                    <AlertCircle size={16} />
                    <div>
                      <strong>{error.message}</strong>
                      {error.details.length > 0 && (
                        <ul>
                          {error.details.slice(0, 5).map((d) => (
                            <li key={d}>{d}</li>
                          ))}
                        </ul>
                      )}
                    </div>
                  </div>
                ) : result ? (
                  <motion.div
                    key={result.response.submission_id}
                    className="stack"
                    initial="hidden"
                    animate="show"
                    variants={{ show: { transition: { staggerChildren: 0.12 } } }}
                  >
                    <div className="results-head">
                      <h2>Assessment results</h2>
                      <span className="mono" style={{ color: "var(--faint)" }}>
                        Mode {result.response.mode} · {new Date(result.response.created_at).toLocaleString()}
                      </span>
                    </div>

                    {result.response.warnings.length > 0 && (
                      <div className="banner banner-warn">
                        <TriangleAlert size={16} />
                        <div>
                          {result.response.warnings.map((w) => (
                            <div key={w}>{w}</div>
                          ))}
                        </div>
                      </div>
                    )}
                    {result.response.input_adjustments.length > 0 && (
                      <div className="banner banner-info">
                        <Info size={16} />
                        <div>
                          {result.response.input_adjustments.map((a) => (
                            <div key={a}>{a}</div>
                          ))}
                        </div>
                      </div>
                    )}

                    {result.response.results.map((r) => (
                      <DiseaseResultCard
                        key={r.config_key}
                        result={r}
                        submitted={result.submitted}
                        labels={labels}
                        units={units}
                      />
                    ))}

                    {result.response.history_saved && (
                      <div className="disclaimer" style={{ color: "var(--neg-ink)" }}>
                        <CheckCircle2 size={16} />
                        <span>Saved to history.</span>
                      </div>
                    )}
                    <DisclaimerNote text={result.response.disclaimer} />
                  </motion.div>
                ) : (
                  <div className="glass empty">
                    <div className="empty-icon" aria-hidden>
                      <ClipboardList size={26} />
                    </div>
                    <h3>Your estimates will appear here</h3>
                    <p>
                      Age, sex and BMI are required. Everything else is optional; adding more information, such as
                      blood pressure or laboratory results, gives the models more to work with.
                    </p>
                    <div className="preview-trio" aria-hidden>
                      <div className="preview-chip">Cardiovascular</div>
                      <div className="preview-chip">Diabetes</div>
                      <div className="preview-chip">Hypertension</div>
                    </div>
                  </div>
                )}
              </section>
            </div>
          </>
        )}

        {tab === "history" && <HistoryView />}
        {tab === "performance" && <ModelPerformanceView />}
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}
