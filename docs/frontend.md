# Frontend

> Next.js frontend, implemented in Phases 3A–3C. It talks only to the `/v2/*` API (see [api-and-database.md](api-and-database.md)); the V1 prototype it replaced has been retired from the whole repository. Status: light redesign applied in Phase 3C; **type-checks and builds cleanly** (`npx tsc --noEmit`, `npm run build`); data shapes were checked against the live backend. A visual/browser pass and end-to-end verification are still outstanding (Phase 4).

## 1. Stack and setup

| Item | Value |
|---|---|
| Framework | Next.js `^16.2.7` (App Router, Turbopack), React 18.3.1, TypeScript 5.5.4 (`strict`) |
| UI libraries | `framer-motion` (animation), `lucide-react` (icons) |
| Styling | Plain global CSS design tokens in `app/globals.css` (no Tailwind / CSS modules) |
| Fonts | Plus Jakarta Sans (headings, UI labels), DM Sans (body, controls), JetBrains Mono (thresholds, metadata); tabular numerals on all numbers |
| API base URL | `NEXT_PUBLIC_API_BASE_URL`, default `http://127.0.0.1:8000` |
| Per-user history | `lib/clientId.ts` keeps a random id in `localStorage` (`hra_client_id`); `lib/api.ts` sends it as `X-Client-ID` on every request |
| Path alias | `@/*` → `frontend/*` (`tsconfig.json`) |

```
cd frontend
npm install
npm run dev        # http://localhost:3000
npm run build      # production build
npx tsc --noEmit   # type check
```

The backend must be running (`uvicorn backend.api.main:app`). Its CORS origins come from `CORS_ORIGINS` (default `localhost:3000` / `127.0.0.1:3000`); set it to the deployed frontend origin when hosting.

**Per-browser history:** the first time the app runs in a browser it generates an id with `crypto.randomUUID()` (random-hex fallback) and stores it under `hra_client_id`. Every API call carries it as `X-Client-ID`, so each browser sees only its own saved assessments, with no login. It is not authentication, and clearing site data or using another browser starts a fresh, empty history.

## 2. Structure

```
frontend/
├── app/
│   ├── layout.tsx        # fonts, metadata ("Health Risk Assessment")
│   ├── page.tsx          # single route "/": tab state, assessment orchestration, sample profiles
│   └── globals.css       # design tokens, glass utilities, all component styles
├── components/
│   ├── Navbar.tsx                 # brand + tabs (Assessment, History, Model Performance), shared-layout pill
│   ├── ModeToggle.tsx             # Mode A / Mode B segmented control (layoutId pill)
│   ├── AssessmentForm.tsx         # backend-driven form, collapsible lab section, validation
│   ├── DiseaseResultCard.tsx      # one condition: decision, probability, gauge, factors, Model Notes
│   ├── ThresholdGauge.tsx         # animated probability bar with a pin at the locked threshold
│   ├── ShapDivergingChart.tsx     # diverging bars (relative magnitude only)
│   ├── ResultsSkeleton.tsx        # shimmer state while /v2/predict runs
│   ├── HistoryView.tsx            # saved assessments: list, detail drawer, delete, clear all
│   └── ModelPerformanceView.tsx   # offline evaluation for all six configurations
└── lib/
    ├── types.ts          # interfaces mirroring backend/api/v2/schemas.py and router responses
    ├── api.ts            # typed client for every /v2 endpoint used + ApiError mapping
    ├── form.ts           # builds request `inputs` from backend field metadata; number formatting
    ├── labels.ts         # display-only labels (education levels, capitalised options)
    └── motion.ts         # shared spring / flow transitions
```

There is one route (`/`); the three tabs are client-side views, not URLs.

## 3. API usage

| Call (in `lib/api.ts`) | Endpoint | Used by |
|---|---|---|
| `getFeatures(mode)` | `GET /v2/features/{mode}` | `page.tsx` (form), `HistoryView` (labels/units for stored inputs) |
| `predict(body)` | `POST /v2/predict` | `page.tsx` |
| `getHistory(limit, beforeId)` | `GET /v2/history?limit&before_id` | `HistoryView` |
| `getHistoryDetail(id)` | `GET /v2/history/{id}` | `HistoryView` |
| `deleteHistoryItem(id)` | `DELETE /v2/history/{id}` | `HistoryView` |
| `clearHistory()` | `DELETE /v2/history?confirm=true` | `HistoryView` |
| `getModelMetrics()` | `GET /v2/model-metrics` | `ModelPerformanceView` |

`ApiError` turns failures into readable messages: network failure → "Cannot reach the assessment service"; 422 → per-field details (the `inputs.` path is stripped); 503 → "models currently unavailable" plus the backend's reasons; 500 → generic message with the request id; string `detail` (e.g. 404) is passed through.

## 4. Views

### 4.1 Assessment

- **Mode A / Mode B** toggle (spring-animated pill). The form is rendered entirely from `GET /v2/features/{mode}`: labels, units, bounds, enum options and required markers come from the backend, so no field list or encoding table is duplicated in TypeScript. Fields are only *grouped* for layout (profile, vitals, lifestyle); any field not in a known group falls into the collapsible **Laboratory biomarkers** section (animated expand/collapse; opens automatically when a sample fills it).
- Required: `age`, `gender`, `bmi`. Everything else is optional; empty fields are omitted from the request and the backend sends them to the model as missing. Client-side checks use the backend's hard bounds; the backend remains authoritative (422 messages are shown too).
- **Quick-fill sample profiles** (lower/higher-risk, per mode) are demonstration data in `page.tsx`; their option strings are the backend's allowed values.
- "Save this assessment to history" (default on) sends `save_to_history`. While it is checked, an animated optional **Patient name** field appears (placeholder "e.g., Aarav Sharma"); the trimmed name is sent as `patient_name` only when saving is on and the field is non-empty.
- **Missing-labs nudge:** in Mode B, when fewer than 4 laboratory fields are filled, a soft-amber, non-blocking banner explains that Mode A gives a more reliable estimate without lab results, with a one-click "Switch to Mode A" button (Phase 2 found Mode B estimates drift lower when labs are omitted).
- **Results**: three independent cards (CVD, Diabetes, Hypertension), staggered entrance, shimmer skeleton while loading. There is **no overall score or composite tier anywhere**.
  - *Decision pill*: "At or above screening threshold" (coral) / "Below screening threshold" (emerald), from `decision_label`.
  - *Estimated probability* with an animated bar and a pin at the locked threshold (adaptive scale so low probabilities stay readable).
  - *What this estimates*: the backend's `target_definition` (e.g. CVD is a self-reported prior diagnosis).
  - *Key Contributing Factors*: a diverging bar chart centred at 0 (right = raises model output, left = lowers it). **Relative magnitude only**: no `%`, no numeric probability, never described as a change in probability. An expandable "How to read this chart" shows `output_space`, `method`, `reference_population` and `notice`. Features the user did not provide are labelled "not provided".
  - *Model Notes* (collapsible, **hidden when empty**): `warnings`, "Provided but not used for this condition" (`ignored_inputs`, e.g. blood pressure for Hypertension, HbA1c/glucose for Diabetes B) and "Missing fields" (`imputed_features`, only when any). Algorithm, version, artifact hash and the locked threshold are no longer shown.
  - *What this estimates* shows the backend definition with survey/guideline codes such as "(NHANES MCQ160B-F)" stripped for display. "How to read this chart" is a plain two-sentence explanation (relative influence within the chart only; not a percentage-point change in probability; not cause and effect); the technical block (output space, method, reference) is not shown.
  - Top-level `warnings`, `input_adjustments` (e.g. age top-coding), a "Saved to history" note and the research-prototype `disclaimer` are shown under the cards.

### 4.2 History

Newest first with cursor pagination ("Load more", using `next_before_id`; duplicates are filtered). Each card shows the **patient name as its title** (or "Unnamed assessment" when none was saved), the local-time timestamp, a `Mode A`/`Mode B` badge and three compact per-condition pills (probability plus decision colour). Expanding a card calls `GET /v2/history/{id}` and shows the patient name, the submitted inputs (labels/units from the feature metadata, enums capitalised, education decoded) and, per condition, probability, locked threshold, decision and number of missing inputs. A single submission is deleted with an inline "Delete this assessment? Delete / Cancel" step (a 404 is treated as already deleted); "Clear all history" uses an inline confirmation and sends `?confirm=true`. Empty and error states (with retry) are handled.

### 4.3 Model Performance

Fetches `GET /v2/model-metrics` and shows, per condition, Mode A and Mode B side by side with a "B − A" column:

- **Discrimination:** ROC-AUC, PR-AUC. **Calibration:** Brier score, ECE.
- **Operating point at the locked threshold:** threshold, sensitivity, specificity, precision, NPV, F1, balanced accuracy.
- Row labels carry plain-language tooltips; the delta column is teal when Mode B is better and amber when worse on that metric (for Brier and ECE lower is better).
- One concise summary card ("Mode A vs Mode B: a trade-off, not an upgrade") whose counts are **computed from the returned numbers** (not hard-coded): how many conditions Mode B improves in ROC-AUC/PR-AUC versus how many conditions Mode A flags more true cases at the screening thresholds, followed by one line noting these are offline research-prototype results, not clinical validation. Mode B is never presented as universally better. The backend `caveats` list is still returned by `/v2/model-metrics` but is no longer rendered as a separate callout.

## 5. Design system

Soft, light-first "alabaster and botanical pastels" (Phase 3C replaced the earlier dark/neon theme). All tokens live in `app/globals.css`; motion constants in `lib/motion.ts`.

- **Canvas:** warm porcelain `#F6F8FB` with two slowly drifting pastel mesh blobs (sage, mist sky, peach/lavender; 22–26 s ease-in-out alternate; disabled under `prefers-reduced-motion`).
- **Surfaces:** frosted white glass (`rgba(255,255,255,0.78)`, `blur(20px) saturate(160%)`, white 1 px border, layered soft shadow); nested wells/inputs are a soft cloud tint that turns white with a teal halo on focus.
- **Ink:** `#1E293B` primary, `#475569` secondary, `#64748B` helper text (never pure black).
- **Semantics (soft, never neon):** botanical teal = primary action/brand; coral/rose on pastel = at or above threshold and *raises model output*; eucalyptus/seafoam on pastel = below threshold and *lowers model output*; periwinkle = Mode B / laboratory accent and Mode badges; amber = model warnings and "worse" deltas.
- **Type:** Plus Jakarta Sans (headings, UI labels, numerals), DM Sans (body and controls), JetBrains Mono (thresholds, technical metadata); `font-variant-numeric: tabular-nums` on probabilities, thresholds and metric tables.
- **Geometry:** cards 28 px radius, nested sections/callouts/drawers 20 px, inputs 16 px, buttons/chips/tabs/badges fully pill-shaped.
- **Motion (Framer Motion):** one soft spring (`stiffness 280, damping 26, mass 0.9`) for the floating navbar pill, Mode A/B toggle, card entrances and tab switches; `whileHover` lift on cards, `whileTap` scale on pills/chips/buttons; `AnimatePresence` height + opacity flow for the laboratory section, Model Notes, chart explanation and History drawers; spring-animated probability and factor bars.
- **Components:** floating frosted pill navbar with centred segmented tabs; threshold gauge is a plump 12 px capsule with a floating pill label above a white/slate marker dot at the locked threshold; the factor chart uses fully rounded bars with row hover highlight.

Language stays plain preventive-screening wording ("Estimated Probability", "Screening Threshold", "Key Contributing Factors").
## 6. Display labels (`lib/labels.ts`)

Display-only; values sent to the API are unchanged. `education_level` is shown as a select (1 Less than 9th grade, 2 9–11th grade, 3 High school / GED, 4 Some college / AA, 5 College graduate or above) but still submits the numeric code 1–5. Enum options (`male`, `never`, `former`, `high`, …) are shown capitalised but submitted as the backend's lowercase strings. The education wording is a frontend presentation of NHANES `DMDEDUC2` codes, not something the API returns.

## 7. Invariants

- No overall/composite score, tier or combined result; the three conditions are always separate.
- Form content is backend-driven; no duplicated encoding tables.
- SHAP is shown as relative magnitude only; never `%`, never probability points, never causal; precomputed research SHAP is never shown as a user's explanation.
- Mode A vs Mode B is presented as a trade-off.
- The research-prototype / non-diagnostic disclaimer from the API is always displayed with results.

## 8. Known limitations and outstanding work

- Not yet browser-tested end to end (Phase 4); only type-check, production build and API-shape checks have been run.
- Single route; tabs are not deep-linkable. No authentication (none exists in the backend).
- Quick-fill profiles are illustrative demo values, not clinical examples.
- Letter-case and wording of option labels other than education is derived by capitalising the backend's strings; the backend supplies no display labels.
- The V1 Streamlit dashboard no longer exists (retired in Phase 5).
