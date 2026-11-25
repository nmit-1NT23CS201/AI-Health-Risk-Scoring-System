"use client";

import { motion } from "framer-motion";
import { formatPercent } from "@/lib/form";

/**
 * Probability track with a pin at the locked screening threshold. The scale adapts so low
 * probabilities stay readable; both end labels state the visible range.
 */
export default function ThresholdGauge({
  probability,
  threshold,
  positive,
}: {
  probability: number;
  threshold: number;
  positive: boolean;
}) {
  const needed = Math.max(probability, threshold) * 1.6;
  const scaleMax = Math.min(1, Math.max(0.25, Math.ceil(needed * 20) / 20));
  const fillPct = Math.min(100, (probability / scaleMax) * 100);
  const pinPct = Math.min(100, (threshold / scaleMax) * 100);

  return (
    <div
      className="gauge"
      role="img"
      aria-label={`Estimated probability ${formatPercent(probability)}; screening threshold ${formatPercent(threshold)}`}
    >
      <div className="gauge-track">
        <div className="gauge-zone" style={{ width: `${100 - pinPct}%` }} />
        <motion.div
          className={`gauge-fill ${positive ? "positive" : "negative"}`}
          initial={{ width: 0 }}
          animate={{ width: `${fillPct}%` }}
          transition={{ type: "spring", stiffness: 70, damping: 18, mass: 1, delay: 0.15 }}
        />
        <div className="gauge-pin" style={{ left: `${pinPct}%` }}>
          <span className="gauge-pin-label">threshold {formatPercent(threshold)}</span>
          <span className="gauge-pin-dot" />
        </div>
      </div>
      <div className="gauge-scale">
        <span>0%</span>
        <span>{Math.round(scaleMax * 100)}%</span>
      </div>
    </div>
  );
}
