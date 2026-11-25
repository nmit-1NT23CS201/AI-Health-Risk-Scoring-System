"use client";

import { motion } from "framer-motion";
import type { Explanation, Inputs } from "@/lib/types";
import { displayValue } from "@/lib/labels";

/**
 * Diverging bar chart centred at 0. Right = raises model output, left = lowers it.
 * Bar length is relative magnitude within this chart only; it is NOT a probability change,
 * so no percentage or numeric probability is ever shown here.
 */
export default function ShapDivergingChart({
  explanation,
  submitted,
  units,
}: {
  explanation: Explanation;
  submitted: Inputs;
  units: Record<string, string | null>;
}) {
  const feats = explanation.top_features;
  const maxAbs = Math.max(...feats.map((f) => Math.abs(f.shap_value)), 1e-12);

  function valueText(name: string, imputed: boolean, inputValue: number | null): string {
    if (imputed) return "not provided";
    const raw = submitted[name];
    const shown = raw !== undefined ? raw : inputValue;
    if (shown === null || shown === undefined) return "";
    return displayValue(name, shown, units[name]);
  }

  return (
    <div>
      <div className="shap-axis" aria-hidden>
        <span />
        <div className="shap-axis-labels">
          <span className="l">← lowers model output</span>
          <span className="r">raises model output →</span>
        </div>
      </div>
      <div className="shap-rows" role="list">
        {feats.map((f, i) => {
          const width = (Math.abs(f.shap_value) / maxAbs) * 50;
          const cls =
            f.direction === "raises_model_output" ? "raises" : f.direction === "lowers_model_output" ? "lowers" : "neutral";
          const text = valueText(f.feature, f.was_imputed, f.input_value);
          return (
            <div
              className="shap-row"
              role="listitem"
              key={f.feature}
              title={`${f.label}: relative influence ${f.direction.replace(/_/g, " ")} (model-output units, not a change in probability)`}
            >
              <div className="shap-name">
                <div className="n">{f.label}</div>
                <div className={`v${f.was_imputed ? " imputed" : ""}`}>{text}</div>
              </div>
              <div className="shap-plot">
                <div className="shap-center" />
                <motion.div
                  className={`shap-bar ${cls}`}
                  initial={cls === "neutral" ? false : { width: 0 }}
                  animate={cls === "neutral" ? undefined : { width: `${width}%` }}
                  transition={{ type: "spring", stiffness: 90, damping: 16, mass: 0.9, delay: 0.2 + i * 0.05 }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
