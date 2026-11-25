"use client";

import { useState } from "react";
import { AnimatePresence, motion, type Variants } from "framer-motion";
import { ChevronDown, ClipboardList, Info, ShieldAlert } from "lucide-react";
import type { DiseaseResult, Inputs } from "@/lib/types";
import { formatPercent } from "@/lib/form";
import ThresholdGauge from "./ThresholdGauge";
import ShapDivergingChart from "./ShapDivergingChart";
import { FLOW, SPRING } from "@/lib/motion";

const DISEASE_TITLE: Record<string, { title: string; sub: string }> = {
  cvd: { title: "Cardiovascular disease", sub: "Heart failure, coronary disease, angina, heart attack, stroke" },
  diabetes: { title: "Diabetes", sub: "Glycaemic status" },
  hypertension: { title: "Hypertension", sub: "High blood pressure" },
};

export const cardVariants: Variants = {
  hidden: { opacity: 0, y: 22 },
  show: { opacity: 1, y: 0, transition: SPRING },
};

function Disclosure({
  icon,
  title,
  badge,
  children,
}: {
  icon: React.ReactNode;
  title: string;
  badge?: React.ReactNode;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className="disclosure">
      <button type="button" className="disclosure-btn" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        <span className="l">
          {icon}
          {title}
          {badge}
        </span>
        <motion.span animate={{ rotate: open ? 180 : 0 }} transition={{ duration: 0.2 }} style={{ display: "inline-flex" }}>
          <ChevronDown size={16} />
        </motion.span>
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={FLOW}
            style={{ overflow: "hidden" }}
          >
            <div className="disclosure-body">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function DiseaseResultCard({
  result,
  submitted,
  labels,
  units,
}: {
  result: DiseaseResult;
  submitted: Inputs;
  labels: Record<string, string>;
  units: Record<string, string | null>;
}) {
  const positive = result.decision_label === "screen_positive";
  const tone = positive ? "positive" : "negative";
  const meta = DISEASE_TITLE[result.disease] ?? { title: result.disease, sub: "" };
  const nameOf = (n: string) => labels[n] ?? n;
  const noteCount = result.ignored_inputs.length + result.warnings.length;
  const expl = result.explanation;

  return (
    <motion.article
      variants={cardVariants}
      whileHover={{ y: -2 }}
      className={`glass disease-card ${tone}`}
    >
      <div className="card-top">
        <div>
          <h3 className="disease-name">{meta.title}</h3>
          <div className="disease-sub">{meta.sub}</div>
        </div>
        <span className={`decision-pill ${tone}`}>
          <span
            aria-hidden
            style={{ width: 8, height: 8, borderRadius: 99, background: positive ? "var(--pos)" : "var(--neg)" }}
          />
          {positive ? "At or above screening threshold" : "Below screening threshold"}
        </span>
      </div>

      <div className="prob-row">
        <div className={`prob-value ${tone}`}>{formatPercent(result.probability)}</div>
        <div className="prob-caption">
          Estimated probability
          <br />
          <span className="mono">threshold {result.threshold.toFixed(2)}</span>
        </div>
      </div>

      <ThresholdGauge probability={result.probability} threshold={result.threshold} positive={positive} />

      <div className="target-def">
        <Info size={15} />
        <span>
          <strong>What this estimates:</strong> {result.target_definition}
        </span>
      </div>

      <div className="shap">
        <div className="shap-head">
          <div className="section-title">Key Contributing Factors</div>
        </div>
        {expl ? (
          <>
            <ShapDivergingChart explanation={expl} submitted={submitted} units={units} />
            <Disclosure icon={<Info size={15} />} title="How to read this chart">
              <div className="note-block">
                <p>
                  Bar length shows relative influence within this chart only. It is not a change in the estimated
                  probability and does not show cause and effect.
                </p>
              </div>
              <div className="note-block">
                <h4>Technical</h4>
                <dl className="meta-grid">
                  <dt>Output space</dt>
                  <dd className="mono">{expl.output_space}</dd>
                  <dt>Method</dt>
                  <dd className="mono">
                    {expl.method} on {expl.explained_estimator}
                  </dd>
                  <dt>Reference</dt>
                  <dd>{expl.reference_population}</dd>
                </dl>
              </div>
              <div className="note-block">
                <p>{expl.notice}</p>
              </div>
            </Disclosure>
          </>
        ) : (
          <div className="banner banner-info" style={{ marginTop: 10 }}>
            <Info size={16} />
            <span>
              Contributing factors are unavailable for this model
              {result.explanation_unavailable_reason ? `: ${result.explanation_unavailable_reason}` : "."}
            </span>
          </div>
        )}
      </div>

      <Disclosure
        icon={<ClipboardList size={15} />}
        title="Model Notes"
        badge={noteCount > 0 ? <span className="count-badge">{noteCount}</span> : undefined}
      >
        {result.warnings.length > 0 && (
          <div className="note-block">
            <h4>Warnings</h4>
            <div className="tag-list">
              {result.warnings.map((w) => (
                <span className="tag warn" key={w}>
                  {w}
                </span>
              ))}
            </div>
          </div>
        )}
        {result.ignored_inputs.length > 0 && (
          <div className="note-block">
            <h4>Provided but not used by this model</h4>
            <div className="tag-list">
              {result.ignored_inputs.map((n) => (
                <span className="tag" key={n}>
                  {nameOf(n)}
                </span>
              ))}
            </div>
            <p style={{ marginTop: 6 }}>
              Some measurements help define the condition being estimated, so the model deliberately excludes them.
            </p>
          </div>
        )}
        <div className="note-block">
          <h4>
            Not provided ({result.imputed_features.length}) — handled by the model&apos;s missing-value rules
          </h4>
          {result.imputed_features.length > 0 ? (
            <div className="tag-list">
              {result.imputed_features.map((n) => (
                <span className="tag" key={n}>
                  {nameOf(n)}
                </span>
              ))}
            </div>
          ) : (
            <p>All inputs used by this model were provided.</p>
          )}
        </div>
        <div className="note-block">
          <h4>Model</h4>
          <dl className="meta-grid">
            <dt>Algorithm</dt>
            <dd>{result.model_name}</dd>
            <dt>Version</dt>
            <dd className="mono">{result.model_version}</dd>
            <dt>Artifact</dt>
            <dd className="mono">{result.artifact_sha256.slice(0, 16)}…</dd>
            <dt>Locked threshold</dt>
            <dd className="mono">{result.threshold.toFixed(2)}</dd>
          </dl>
          <p style={{ marginTop: 8 }}>{result.probability_note}</p>
        </div>
      </Disclosure>
    </motion.article>
  );
}

export function DisclaimerNote({ text }: { text: string }) {
  return (
    <div className="disclaimer">
      <ShieldAlert size={16} />
      <span>{text}</span>
    </div>
  );
}
