"use client";

import { motion } from "framer-motion";
import { SPRING, TAP } from "@/lib/motion";
import { BarChart3, ClipboardList, HeartPulse, History } from "lucide-react";

export type TabId = "assessment" | "history" | "performance";

const TABS: { id: TabId; label: string; icon: typeof ClipboardList }[] = [
  { id: "assessment", label: "Assessment", icon: ClipboardList },
  { id: "history", label: "History", icon: History },
  { id: "performance", label: "Model Performance", icon: BarChart3 },
];

export default function Navbar({ tab, onChange }: { tab: TabId; onChange: (t: TabId) => void }) {
  return (
    <header className="navbar">
      <div className="container navbar-inner">
        <div className="brand">
          <div className="brand-mark" aria-hidden>
            <HeartPulse size={20} strokeWidth={2.2} />
          </div>
          <div>
            <div className="brand-name">Health Risk Assessment</div>
            <div className="brand-sub">Preventive Screening Support</div>
          </div>
        </div>

        <nav className="tabs" role="tablist" aria-label="Sections">
          {TABS.map(({ id, label, icon: Icon }) => (
            <motion.button
              key={id}
              whileTap={TAP}
              role="tab"
              aria-selected={tab === id}
              className="tab"
              onClick={() => onChange(id)}
            >
              {tab === id && (
                <motion.span
                  layoutId="nav-pill"
                  className="tab-pill"
                  transition={SPRING}
                />
              )}
              <span>
                <Icon size={16} />
                <span className="tab-label">{label}</span>
              </span>
            </motion.button>
          ))}
        </nav>
        <div className="navbar-spacer" />
      </div>
    </header>
  );
}
