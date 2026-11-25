"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { AlertCircle, BarChart3, RotateCcw, Scale } from "lucide-react";
import { getModelMetrics } from "@/lib/api";
import { SPRING } from "@/lib/motion";
import type { Disease, ModelConfigMetrics, ModelMetricsResponse } from "@/lib/types";

const DISEASES: { id: Disease; title: string; sub: string }[] = [
  { id: "cvd", title: "Cardiovascular disease", sub: "Self-reported prior diagnosis" },
  { id: "diabetes", title: "Diabetes", sub: "Measured values or diagnosis" },
  { id: "hypertension", title: "Hypertension", sub: "Measured blood pressure or diagnosis" },
];

type Row = {
  key: string;
  label: string;
  hint: string;
  get: (c: ModelConfigMetrics) => number;
  better: "higher" | "lower" | "none";
  digits?: number;
};

const GROUPS: { title: string; rows: Row[] }[] = [
  {
    title: "Discrimination",
    rows: [
      { key: "roc", label: "ROC-AUC", hint: "How well the model ranks people with the condition above those without (0.5 = chance, 1 = perfect).", get: (c) => c.test_metrics.roc_auc, better: "higher" },
      { key: "pr", label: "PR-AUC", hint: "Precision–recall trade-off across all thresholds; depends strongly on how common the condition is.", get: (c) => c.test_metrics.pr_auc, better: "higher" },
    ],
  },
  {
    title: "Calibration",
    rows: [
      { key: "brier", label: "Brier score", hint: "Mean squared error of the estimated probabilities (lower is better).", get: (c) => c.test_metrics.brier_score, better: "lower" },
      { key: "ece", label: "ECE", hint: "Expected calibration error: average gap between estimated and observed rates (lower is better).", get: (c) => c.test_metrics.ece, better: "lower" },
    ],
  },
  {
    title: "Operating point at the locked screening threshold",
    rows: [
      { key: "thr", label: "Threshold", hint: "Locked cut-off chosen on validation data; estimates at or above it are flagged.", get: (c) => c.locked_threshold, better: "none", digits: 2 },
      { key: "sens", label: "Sensitivity", hint: "Share of people with the condition who are flagged.", get: (c) => c.test_binary_at_locked_threshold.sensitivity, better: "higher" },
      { key: "spec", label: "Specificity", hint: "Share of people without the condition who are not flagged.", get: (c) => c.test_binary_at_locked_threshold.specificity, better: "higher" },
      { key: "prec", label: "Precision", hint: "Share of flagged people who have the condition.", get: (c) => c.test_binary_at_locked_threshold.precision, better: "higher" },
      { key: "npv", label: "NPV", hint: "Share of people not flagged who do not have the condition.", get: (c) => c.test_binary_at_locked_threshold.npv, better: "higher" },
      { key: "f1", label: "F1", hint: "Balance of precision and sensitivity.", get: (c) => c.test_binary_at_locked_threshold.f1, better: "higher" },
      { key: "bacc", label: "Balanced accuracy", hint: "Average of sensitivity and specificity.", get: (c) => c.test_binary_at_locked_threshold.balanced_accuracy, better: "higher" },
    ],
  },
];

const fmt = (n: number, d = 4) => n.toFixed(d);

function Delta({ row, a, b }: { row: Row; a: number; b: number }) {
  if (row.better === "none") return <span className="delta flat">—</span>;
  const d = b - a;
  if (Math.abs(d) < 0.00005) return <span className="delta flat">0.0000</span>;
  const improved = row.better === "higher" ? d > 0 : d < 0;
  return (
    <span className={`delta ${improved ? "up" : "down"}`} title={improved ? "Mode B is better on this metric" : "Mode A is better on this metric"}>
      {d > 0 ? "+" : "−"}
      {Math.abs(d).toFixed(4)}
    </span>
  );
}

function DiseasePanel({
  meta,
  a,
  b,
}: {
  meta: (typeof DISEASES)[number];
  a: ModelConfigMetrics;
  b: ModelConfigMetrics;
}) {
  return (
    <motion.article
      className="glass perf-card"
      initial={{ opacity: 0, y: 18 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-40px" }}
      whileHover={{ y: -2 }}
      transition={SPRING}
    >
      <div className="card-top">
        <div>
          <h3 className="disease-name">{meta.title}</h3>
          <div className="disease-sub">{meta.sub}</div>
        </div>
      </div>

      <div className="perf-scroll">
        <table className="perf-table">
          <thead>
            <tr>
              <th scope="col" />
              <th scope="col">
                <span className="mode-badge">Mode A</span>
                <small>{a.model_name}</small>
                <small className="mono">n={a.n_test.toLocaleString()}</small>
              </th>
              <th scope="col">
                <span className="mode-badge">Mode B</span>
                <small>{b.model_name}</small>
                <small className="mono">n={b.n_test.toLocaleString()}</small>
              </th>
              <th scope="col" className="dcol">
                B − A
              </th>
            </tr>
          </thead>
          <tbody>
            {GROUPS.map((g) => (
              <PerfGroup key={g.title} title={g.title} rows={g.rows} a={a} b={b} />
            ))}
          </tbody>
        </table>
      </div>
    </motion.article>
  );
}

function PerfGroup({ title, rows, a, b }: { title: string; rows: Row[]; a: ModelConfigMetrics; b: ModelConfigMetrics }) {
  return (
    <>
      <tr className="group-row">
        <th colSpan={4} scope="colgroup">
          {title}
        </th>
      </tr>
      {rows.map((r) => {
        const av = r.get(a);
        const bv = r.get(b);
        const d = r.digits ?? 4;
        return (
          <tr key={r.key}>
            <th scope="row" title={r.hint}>
              {r.label}
            </th>
            <td className="num">{fmt(av, d)}</td>
            <td className="num">{fmt(bv, d)}</td>
            <td className="num dcol">
              <Delta row={r} a={av} b={bv} />
            </td>
          </tr>
        );
      })}
    </>
  );
}

export default function ModelPerformanceView() {
  const [data, setData] = useState<ModelMetricsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await getModelMetrics());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load evaluation results.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const pairs = useMemo(() => {
    if (!data) return null;
    const find = (d: Disease, m: "A" | "B") => Object.values(data.configs).find((c) => c.disease === d && c.mode === m);
    const out = DISEASES.map((meta) => ({ meta, a: find(meta.id, "A"), b: find(meta.id, "B") }));
    return out.every((p) => p.a && p.b) ? (out as { meta: (typeof DISEASES)[number]; a: ModelConfigMetrics; b: ModelConfigMetrics }[]) : null;
  }, [data]);

  // Trade-off summary derived from the numbers themselves, never asserted.
  const summary = useMemo(() => {
    if (!pairs) return null;
    const total = pairs.length;
    const bAuc = pairs.filter((p) => p.b.test_metrics.roc_auc > p.a.test_metrics.roc_auc).length;
    const bPr = pairs.filter((p) => p.b.test_metrics.pr_auc > p.a.test_metrics.pr_auc).length;
    const aSens = pairs.filter(
      (p) => p.a.test_binary_at_locked_threshold.sensitivity > p.b.test_binary_at_locked_threshold.sensitivity,
    ).length;
    const aBacc = pairs.filter(
      (p) => p.a.test_binary_at_locked_threshold.balanced_accuracy > p.b.test_binary_at_locked_threshold.balanced_accuracy,
    ).length;
    return { total, bAuc, bPr, aSens, aBacc };
  }, [pairs]);

  return (
    <>
      <div className="page-head">
        <div className="eyebrow">Model performance</div>
        <h1>How the models were evaluated</h1>
        <p>
          Held-out test results for each condition in both modes. Each condition has its own model; nothing here is
          combined into a single score.
        </p>
      </div>

      {error && (
        <div className="banner banner-error" role="alert">
          <AlertCircle size={16} />
          <div>
            <strong>{error}</strong>
            <div>
              <button type="button" className="btn-ghost" style={{ marginTop: 8 }} onClick={() => void load()}>
                <RotateCcw size={13} /> Try again
              </button>
            </div>
          </div>
        </div>
      )}

      {loading && (
        <div className="stack" aria-busy>
          {[0, 1].map((i) => (
            <div className="glass perf-card" key={i}>
              <div className="skeleton sk-title" />
              {[0, 1, 2, 3, 4].map((r) => (
                <div className="skeleton" key={r} style={{ height: 16, marginTop: 12 }} />
              ))}
            </div>
          ))}
        </div>
      )}

      {data && pairs && summary && (
        <div className="stack">
          <section className="glass callout" aria-label="Methodology and limitations">
            <div className="callout-head">
              <Scale size={18} />
              <h2>Read these results with these limits in mind</h2>
            </div>
            <ul className="callout-list">
              {data.caveats.map((c) => (
                <li key={c}>{c}</li>
              ))}
              <li>
                These are research results for a prototype, not clinical validation. The screening thresholds were
                fixed in advance on separate validation data.
              </li>
            </ul>
          </section>

          <section className="glass callout tradeoff" aria-label="Mode A versus Mode B">
            <div className="callout-head">
              <BarChart3 size={18} />
              <h2>Mode A vs Mode B: a trade-off, not an upgrade</h2>
            </div>
            <p>
              Adding laboratory results (Mode B) raised ROC-AUC in {summary.bAuc} of {summary.total} conditions and
              PR-AUC in {summary.bPr} of {summary.total}. At the locked screening thresholds, however, Mode A had the
              higher sensitivity in {summary.aSens} of {summary.total} conditions and the higher balanced accuracy in{" "}
              {summary.aBacc} of {summary.total}. The two modes operate at different thresholds, so better ranking does
              not automatically mean more people with the condition are flagged. Which mode suits a given situation
              depends on whether catching more cases or limiting false alarms matters more.
            </p>
          </section>

          {pairs.map((p) => (
            <DiseasePanel key={p.meta.id} meta={p.meta} a={p.a} b={p.b} />
          ))}

          <p className="disclaimer-line">
            Teal and amber in the “B − A” column show whether Mode B is better or worse than Mode A on that metric
            (for Brier score and ECE, lower is better). Evaluation: {data.evaluation.replace(/_/g, " ")}.
          </p>
        </div>
      )}
    </>
  );
}
