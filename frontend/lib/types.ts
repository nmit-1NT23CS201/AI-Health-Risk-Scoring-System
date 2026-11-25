// Types mirror backend/api/v2/schemas.py and the /v2/features/{mode} response (backend/api/v2/router.py).

export type Mode = "A" | "B";
export type Disease = "cvd" | "diabetes" | "hypertension";

export const DISEASES: Disease[] = ["cvd", "diabetes", "hypertension"];

export interface FeatureMeta {
  name: string;
  label: string;
  unit: string | null;
  required: boolean;
  allowed_values: string[] | null;
  hard_bounds: { min: number | null; max: number | null };
  training_range: [number, number] | null;
  description: string | null;
  used_by: Record<Disease, boolean>;
}

export interface FeaturesResponse {
  mode: Mode;
  fields: FeatureMeta[];
}

export type InputValue = string | number;
export type Inputs = Record<string, InputValue>;

export interface PredictRequest {
  mode: Mode;
  diseases?: Disease[];
  inputs: Inputs;
  include_explanation?: boolean;
  save_to_history?: boolean;
  patient_name?: string | null;
}

export interface FeatureAttribution {
  feature: string;
  label: string;
  input_value: number | null;
  was_imputed: boolean;
  shap_value: number;
  direction: "raises_model_output" | "lowers_model_output" | "neutral";
  rank: number;
}

export interface Explanation {
  method: "TreeExplainer" | "LinearExplainer";
  explained_estimator: string;
  output_space: "uncalibrated_rf_probability" | "log_odds_margin";
  base_value: number;
  sum_of_attributions: number;
  base_model_output: number;
  additive_to_calibrated_probability: false;
  calibrated_probability: number;
  top_features: FeatureAttribution[];
  other_features_sum: number;
  reference_population: string;
  display_hint: "relative_magnitude_only";
  notice: string;
}

export interface DiseaseResult {
  disease: Disease;
  mode: Mode;
  config_key: string;
  model_name: string;
  probability: number;
  threshold: number;
  predicted_positive: boolean;
  decision_label: "screen_positive" | "screen_negative";
  target_definition: string;
  probability_note: string;
  imputed_features: string[];
  ignored_inputs: string[];
  warnings: string[];
  model_version: string;
  artifact_sha256: string;
  explanation: Explanation | null;
  explanation_unavailable_reason: string | null;
}

export interface PredictResponse {
  submission_id: string;
  created_at: string;
  mode: Mode;
  results: DiseaseResult[];
  history_saved: boolean;
  input_adjustments: string[];
  warnings: string[];
  disclaimer: string;
}


// ---- /v2/history* (backend/api/v2/schemas.py: History*) ----
export interface HistoryResult {
  disease: Disease;
  config_key: string;
  probability: number;
  threshold: number;
  predicted_positive: boolean;
  model_version: string;
}

export interface HistorySubmission {
  submission_id: string;
  patient_name: string | null;
  created_at: string;
  mode: Mode;
  results: HistoryResult[];
}

export interface HistoryList {
  items: HistorySubmission[];
  next_before_id: number | null;
}

export interface HistoryDetailResult extends HistoryResult {
  model_input: Record<string, number | null>;
  imputed_features: string[];
  artifact_sha256: string;
}

export interface HistoryDetail {
  submission_id: string;
  patient_name: string | null;
  created_at: string;
  mode: Mode;
  inputs: Inputs;
  results: HistoryDetailResult[];
}

export interface HistoryDeleteResponse {
  deleted_rows: number;
  submission_id: string | null;
}

// ---- /v2/model-metrics (backend/api/v2/router.py) ----
export interface ModelTestMetrics {
  roc_auc: number;
  pr_auc: number;
  brier_score: number;
  ece: number;
  mce: number;
  mean_pred_prob: number;
  observed_rate: number;
}

export interface ModelOperatingPoint {
  threshold: number;
  sensitivity: number;
  specificity: number;
  precision: number;
  npv: number;
  accuracy: number;
  balanced_accuracy: number;
  f1: number;
  f2: number;
  youden_j: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
}

export interface ModelConfigMetrics {
  disease: Disease;
  mode: Mode;
  model_name: string;
  locked_threshold: number;
  calibration: string;
  n_train: number;
  n_val: number;
  n_test: number;
  test_metrics: ModelTestMetrics;
  test_binary_at_locked_threshold: ModelOperatingPoint;
}

export interface ModelMetricsResponse {
  evaluation: string;
  caveats: string[];
  configs: Record<string, ModelConfigMetrics>;
}