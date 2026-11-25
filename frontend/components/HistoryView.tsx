"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { AlertCircle, ChevronDown, History as HistoryIcon, Loader2, RotateCcw, Trash2 } from "lucide-react";
import {
  ApiError,
  clearHistory,
  deleteHistoryItem,
  getFeatures,
  getHistory,
  getHistoryDetail,
} from "@/lib/api";
import { formatPercent } from "@/lib/form";
import { FLOW, SPRING, TAP } from "@/lib/motion";
import { displayValue } from "@/lib/labels";
import type { Disease, FeatureMeta, HistoryDetail, HistorySubmission, Mode } from "@/lib/types";

const PAGE_SIZE = 10;

const DISEASE_NAME: Record<Disease, string> = {
  cvd: "Cardiovascular",
  diabetes: "Diabetes",
  hypertension: "Hypertension",
};

// Field metadata (labels, units) per mode, fetched once and shared across rows.
const metaCache = new Map<Mode, Promise<FeatureMeta[]>>();
function loadMeta(mode: Mode): Promise<FeatureMeta[]> {
  let p = metaCache.get(mode);
  if (!p) {
    p = getFeatures(mode)
      .then((r) => r.fields)
      .catch((e) => {
        metaCache.delete(mode);
        throw e;
      });
    metaCache.set(mode, p);
  }
  return p;
}

function formatWhen(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function messageOf(e: unknown): string {
  return e instanceof ApiError || e instanceof Error ? e.message : "Something went wrong.";
}

function DiseasePill({ disease, probability, positive }: { disease: Disease; probability: number; positive: boolean }) {
  return (
    <span className={`hist-pill ${positive ? "positive" : "negative"}`}>
      <span className="dot" aria-hidden />
      <span className="d">{DISEASE_NAME[disease]}</span>
      <span className="p num">{formatPercent(probability)}</span>
      <span className="sr-only">{positive ? "at or above screening threshold" : "below screening threshold"}</span>
    </span>
  );
}

function DetailPanel({ detail, meta }: { detail: HistoryDetail; meta: FeatureMeta[] | null }) {
  const byName = new Map((meta ?? []).map((f) => [f.name, f]));
  const entries = Object.entries(detail.inputs);
  return (
    <div className="hist-detail">
      <div className="note-block">
        <h4>Patient</h4>
        <p className="hist-detail-name">
          {detail.patient_name || "Unnamed assessment"}
          <span className="hist-when"> · {formatWhen(detail.created_at)}</span>
        </p>
      </div>

      <div className="note-block">
        <h4>Submitted inputs</h4>
        <dl className="inputs-grid">
          {entries.map(([name, value]) => {
            const f = byName.get(name);
            return (
              <div key={name} className="kv">
                <dt>{f?.label ?? name.replace(/_/g, " ")}</dt>
                <dd className="num">{displayValue(name, value, f?.unit)}</dd>
              </div>
            );
          })}
        </dl>
      </div>

      <div className="note-block">
        <h4>Decisions at the locked screening thresholds</h4>
        <div className="hist-results">
          {detail.results.map((r) => (
            <div key={r.config_key} className={`hist-result ${r.predicted_positive ? "positive" : "negative"}`}>
              <div className="top">
                <strong>{DISEASE_NAME[r.disease]}</strong>
                <span className={`decision-pill ${r.predicted_positive ? "positive" : "negative"}`}>
                  {r.predicted_positive ? "At or above threshold" : "Below threshold"}
                </span>
              </div>
              <dl className="meta-grid">
                <dt>Estimated probability</dt>
                <dd className="num">{formatPercent(r.probability)}</dd>
                <dt>Screening threshold</dt>
                <dd className="mono">{r.threshold.toFixed(2)}</dd>
                <dt>Not provided</dt>
                <dd>{r.imputed_features.length} inputs</dd>
                <dt>Model</dt>
                <dd className="mono">{r.model_version}</dd>
              </dl>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function SubmissionCard({
  item,
  open,
  onToggle,
  onDeleted,
}: {
  item: HistorySubmission;
  open: boolean;
  onToggle: () => void;
  onDeleted: (id: string) => void;
}) {
  const [detail, setDetail] = useState<HistoryDetail | null>(null);
  const [meta, setMeta] = useState<FeatureMeta[] | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  useEffect(() => {
    if (!open || detail) return;
    let cancelled = false;
    setLoading(true);
    setDetailError(null);
    Promise.all([getHistoryDetail(item.submission_id), loadMeta(item.mode).catch(() => null)])
      .then(([d, m]) => {
        if (cancelled) return;
        setDetail(d);
        setMeta(m);
      })
      .catch((e) => !cancelled && setDetailError(messageOf(e)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [open, detail, item.submission_id, item.mode]);

  async function remove() {
    setDeleting(true);
    setDeleteError(null);
    try {
      await deleteHistoryItem(item.submission_id);
      onDeleted(item.submission_id);
    } catch (e) {
      // Already gone elsewhere: treat as deleted so the list stays truthful.
      if (e instanceof ApiError && e.status === 404) onDeleted(item.submission_id);
      else {
        setDeleteError(messageOf(e));
        setDeleting(false);
      }
    }
  }

  return (
    <motion.article
      layout="position"
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.97 }}
      whileHover={{ y: -2 }}
      transition={SPRING}
      className="glass hist-card"
    >
      <div className="hist-top">
        <button type="button" className="hist-main" aria-expanded={open} onClick={onToggle}>
          <div className="hist-meta">
            <span className={`hist-title${item.patient_name ? "" : " unnamed"}`}>
              {item.patient_name || "Unnamed assessment"}
            </span>
            <span className="mode-badge">Mode {item.mode}</span>
          </div>
          <div className="hist-when">{formatWhen(item.created_at)}</div>
          <div className="hist-pills">
            {item.results.map((r) => (
              <DiseasePill key={r.config_key} disease={r.disease} probability={r.probability} positive={r.predicted_positive} />
            ))}
          </div>
        </button>

        <div className="hist-actions">
          {confirming ? (
            <div className="confirm-inline" role="group" aria-label="Confirm delete">
              <span>Delete this assessment?</span>
              <button type="button" className="btn-danger" onClick={remove} disabled={deleting}>
                {deleting ? <Loader2 size={13} className="spin" /> : "Delete"}
              </button>
              <button type="button" className="btn-ghost" onClick={() => setConfirming(false)} disabled={deleting}>
                Cancel
              </button>
            </div>
          ) : (
            <button type="button" className="icon-btn" aria-label="Delete assessment" onClick={() => setConfirming(true)}>
              <Trash2 size={16} />
            </button>
          )}
          <button type="button" className="icon-btn" aria-label={open ? "Hide details" : "Show details"} onClick={onToggle}>
            <motion.span animate={{ rotate: open ? 180 : 0 }} style={{ display: "inline-flex" }}>
              <ChevronDown size={18} />
            </motion.span>
          </button>
        </div>
      </div>

      {deleteError && (
        <div className="banner banner-error" role="alert" style={{ marginTop: 12 }}>
          <AlertCircle size={16} />
          <span>{deleteError}</span>
        </div>
      )}

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={FLOW}
            style={{ overflow: "hidden" }}
          >
            <div className="hist-drawer">
              {loading && (
                <div className="stack" aria-busy>
                  <div className="skeleton" style={{ height: 14, width: "40%" }} />
                  <div className="skeleton" style={{ height: 70 }} />
                </div>
              )}
              {detailError && (
                <div className="banner banner-error" role="alert">
                  <AlertCircle size={16} />
                  <span>{detailError}</span>
                </div>
              )}
              {detail && <DetailPanel detail={detail} meta={meta} />}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.article>
  );
}

export default function HistoryView() {
  const [items, setItems] = useState<HistorySubmission[]>([]);
  const [cursor, setCursor] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [openId, setOpenId] = useState<string | null>(null);
  const [confirmClear, setConfirmClear] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const alive = useRef(true);

  const loadFirst = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const page = await getHistory(PAGE_SIZE);
      if (!alive.current) return;
      setItems(page.items);
      setCursor(page.next_before_id);
    } catch (e) {
      if (alive.current) setError(messageOf(e));
    } finally {
      if (alive.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    alive.current = true;
    void loadFirst();
    return () => {
      alive.current = false;
    };
  }, [loadFirst]);

  async function loadMore() {
    if (cursor === null) return;
    setLoadingMore(true);
    setError(null);
    try {
      const page = await getHistory(PAGE_SIZE, cursor);
      setItems((prev) => {
        const seen = new Set(prev.map((i) => i.submission_id));
        return [...prev, ...page.items.filter((i) => !seen.has(i.submission_id))];
      });
      setCursor(page.next_before_id);
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setLoadingMore(false);
    }
  }

  async function clearAll() {
    setClearing(true);
    setError(null);
    try {
      const res = await clearHistory();
      setItems([]);
      setCursor(null);
      setOpenId(null);
      setNotice(`Cleared ${res.deleted_rows > 0 ? "all saved assessments" : "history"}.`);
      setConfirmClear(false);
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setClearing(false);
    }
  }

  return (
    <>
      <div className="page-head">
        <div className="eyebrow">History</div>
        <h1>Past assessments</h1>
        <p>
          Assessments you chose to save, newest first. Each one lists the three conditions separately — there is no
          combined score.
        </p>
      </div>

      <div className="hist-toolbar">
        <span className="mono" style={{ color: "var(--faint)" }}>
          {loading ? "Loading…" : `${items.length} shown${cursor !== null ? " · more available" : ""}`}
        </span>
        {items.length > 0 &&
          (confirmClear ? (
            <div className="confirm-inline" role="group" aria-label="Confirm clear all">
              <span>Permanently delete all saved assessments?</span>
              <button type="button" className="btn-danger" onClick={clearAll} disabled={clearing}>
                {clearing ? <Loader2 size={13} className="spin" /> : "Yes, clear all"}
              </button>
              <button type="button" className="btn-ghost" onClick={() => setConfirmClear(false)} disabled={clearing}>
                Cancel
              </button>
            </div>
          ) : (
            <button type="button" className="btn-ghost" onClick={() => setConfirmClear(true)}>
              <Trash2 size={14} /> Clear all history
            </button>
          ))}
      </div>

      {notice && items.length === 0 && (
        <div className="banner banner-info" style={{ marginBottom: 14 }}>
          <HistoryIcon size={16} />
          <span>{notice}</span>
        </div>
      )}

      {error && (
        <div className="banner banner-error" role="alert" style={{ marginBottom: 14 }}>
          <AlertCircle size={16} />
          <div>
            <strong>{error}</strong>
            <div>
              <button type="button" className="btn-ghost" style={{ marginTop: 8 }} onClick={() => void loadFirst()}>
                <RotateCcw size={13} /> Try again
              </button>
            </div>
          </div>
        </div>
      )}

      {loading ? (
        <div className="stack" aria-busy>
          {[0, 1, 2].map((i) => (
            <div key={i} className="glass hist-card">
              <div className="skeleton" style={{ height: 14, width: "30%" }} />
              <div className="skeleton" style={{ height: 30, marginTop: 14 }} />
            </div>
          ))}
        </div>
      ) : items.length === 0 && !error ? (
        <div className="glass empty">
          <div className="empty-icon" aria-hidden>
            <HistoryIcon size={26} />
          </div>
          <h3>No saved assessments yet</h3>
          <p>
            Run an assessment with “Save this assessment to history” selected and it will appear here for later
            review.
          </p>
        </div>
      ) : (
        <div className="stack">
          <AnimatePresence initial={false} mode="popLayout">
            {items.map((item) => (
              <SubmissionCard
                key={item.submission_id}
                item={item}
                open={openId === item.submission_id}
                onToggle={() => setOpenId((cur) => (cur === item.submission_id ? null : item.submission_id))}
                onDeleted={(id) => {
                  setItems((prev) => prev.filter((i) => i.submission_id !== id));
                  setOpenId((cur) => (cur === id ? null : cur));
                }}
              />
            ))}
          </AnimatePresence>
          {cursor !== null && (
            <div style={{ display: "flex", justifyContent: "center" }}>
              <button type="button" className="btn-ghost" onClick={loadMore} disabled={loadingMore}>
                {loadingMore ? (
                  <>
                    <Loader2 size={14} className="spin" /> Loading…
                  </>
                ) : (
                  "Load more"
                )}
              </button>
            </div>
          )}
        </div>
      )}
    </>
  );
}
