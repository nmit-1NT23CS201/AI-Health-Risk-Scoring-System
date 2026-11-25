"use client";

import { motion } from "framer-motion";
import type { Mode } from "@/lib/types";
import { SPRING, TAP } from "@/lib/motion";

const OPTIONS: { id: Mode; title: string; sub: string }[] = [
  { id: "A", title: "Basic", sub: "Profile, vitals, lifestyle" },
  { id: "B", title: "Extended + Labs", sub: "Adds blood biomarkers" },
];

export default function ModeToggle({ mode, onChange }: { mode: Mode; onChange: (m: Mode) => void }) {
  return (
    <div className="segmented" role="radiogroup" aria-label="Assessment mode">
      {OPTIONS.map((o) => (
        <motion.button
          key={o.id}
          whileTap={TAP}
          type="button"
          role="radio"
          aria-checked={mode === o.id}
          className="segment"
          onClick={() => onChange(o.id)}
        >
          {mode === o.id && (
            <motion.span
              layoutId="mode-pill"
              className="segment-pill"
              transition={SPRING}
            />
          )}
          <span className="segment-body">
            <span className="segment-title">
              Mode {o.id} · {o.title}
            </span>
            <span className="segment-sub">{o.sub}</span>
          </span>
        </motion.button>
      ))}
    </div>
  );
}
