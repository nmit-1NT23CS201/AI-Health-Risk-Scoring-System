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

/** Strip survey / guideline codes such as "(NHANES MCQ160B-F)" or "(JNC7-style, NHANES)" for display. */
function plainDefinition(text: string): string {
  return text.replace(/\s*\([^)]*(NHANES|JNC7)[^)]*\)/g, "").replace(/\s+\./g, ".");
}

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
  const noteCount =
    result.warnings.length + (result.ignored_inputs.length > 0 ? 1 : 0) + (result.imputed_features.length > 0 ? 1 : 0);
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
          <strong>What this estimates:</strong> {plainDefinition(result.target_definition)}
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
              <p>
                Bar length shows relative influence within this chart only. It is not a percentage-point change in the
                estimated probability and does not show cause and effect.
              </p>
            </Disclosure>
          </>
        ) : (
          <div className="banner banner-info" style={{ marginTop: 10 }}>
            <Info size={16} />
            <span>Contributing factors are not available for this estimate.</span>
          </div>
        )}
      </div>

      {noteCount > 0 && (
        <Disclosure
          icon={<ClipboardList size={15} />}
          title="Model Notes"
          badge={<span className="count-badge">{noteCount}</span>}
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
              <h4>Provided but not used for this condition</h4>
              <div className="tag-list">
                {result.ignored_inputs.map((n) => (
                  <span className="tag" key={n}>
                    {nameOf(n)}
                  </span>
                ))}
              </div>
              <p style={{ marginTop: 6 }}>
                Some measurements help define the condition itself, so they are deliberately left out of its estimate.
              </p>
            </div>
          )}
          {result.imputed_features.length > 0 && (
            <div className="note-block">
              <h4>Missing fields ({result.imputed_features.length})</h4>
              <div className="tag-list">
                {result.imputed_features.map((n) => (
                  <span className="tag" key={n}>
                    {nameOf(n)}
                  </span>
                ))}
              </div>
              <p style={{ marginTop: 6 }}>These were left blank, so the estimate relies on the information provided.</p>
            </div>
          )}
        </Disclosure>
      )}
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
