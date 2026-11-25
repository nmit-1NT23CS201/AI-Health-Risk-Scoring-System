import type { Transition } from "framer-motion";

/** Soft, organic spring used for pills, cards and toggles. */
export const SPRING: Transition = { type: "spring", stiffness: 280, damping: 26, mass: 0.9 };

/** Height/opacity flow for drawers and collapsible sections. */
export const FLOW: Transition = { type: "spring", stiffness: 260, damping: 30, mass: 0.9 };

export const TAP = { scale: 0.97 };
