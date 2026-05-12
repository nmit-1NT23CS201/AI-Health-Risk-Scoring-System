"use client";

import { useMemo, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

type PredictionInput = {
  age: number;
  gender: string;
  bmi: number;
  systolic_bp: number;
  diastolic_bp: number;
  cholesterol_mg_dl: number;
  smoking: string;
  alcohol_consumption: string;
  physical_activity: string;
  family_history: string;
  heart_rate_bpm: number;
  sdnn_hrv: number;
  rmssd_hrv: number;
  spo2: number;
};

type Contributor = {
  feature: string;
  value: number;
};

type PredictionResponse = {
  risk_score: number;
  risk_level: string;
  positive_contributors: Contributor[];
  negative_contributors: Contributor[];
  insights: string[];
  recommendations: string[];
  summary_plot: string;
  bar_plot: string;
  waterfall_plot: string;
};

const defaultValues: PredictionInput = {
  age: 40,
  gender: "Male",
  bmi: 25,
  systolic_bp: 125,
  diastolic_bp: 80,
  cholesterol_mg_dl: 190,
  smoking: "No",
  alcohol_consumption: "No",
  physical_activity: "Medium",
  family_history: "No",
  heart_rate_bpm: 76,
  sdnn_hrv: 55,
  rmssd_hrv: 50,
  spo2: 97,
};

const highRiskPreset: PredictionInput = {
  age: 55,
  gender: "Male",
  bmi: 31,
  systolic_bp: 145,
  diastolic_bp: 92,
  cholesterol_mg_dl: 250,
  smoking: "Yes",
  alcohol_consumption: "Yes",
  physical_activity: "Low",
  family_history: "Yes",
  heart_rate_bpm: 92,
  sdnn_hrv: 35,
  rmssd_hrv: 28,
  spo2: 94,
};

const lowRiskPreset: PredictionInput = {
  age: 24,
  gender: "Female",
  bmi: 22,
  systolic_bp: 112,
  diastolic_bp: 72,
  cholesterol_mg_dl: 160,
  smoking: "No",
  alcohol_consumption: "No",
  physical_activity: "High",
  family_history: "No",
  heart_rate_bpm: 68,
  sdnn_hrv: 78,
  rmssd_hrv: 65,
  spo2: 99,
};

function riskPillClass(riskLevel: string) {
  if (riskLevel === "Low Risk") {
    return "risk-pill pill-low";
  }
  if (riskLevel === "Medium Risk") {
    return "risk-pill pill-medium";
  }
  return "risk-pill pill-high";
}

export default function Home() {
  const [formValues, setFormValues] = useState<PredictionInput>(defaultValues);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const riskPercent = useMemo(() => {
    if (!prediction) return 0;
    return Math.max(0, Math.min(100, prediction.risk_score));
  }, [prediction]);

  const predictionState = prediction ? "Prediction Complete" : "Awaiting Input";

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formValues),
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || "Prediction failed.");
      }

      const payload = (await response.json()) as PredictionResponse;
      setPrediction(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed.");
    } finally {
      setLoading(false);
    }
  }

  function updateField<K extends keyof PredictionInput>(
    key: K,
    value: PredictionInput[K]
  ) {
    setFormValues((prev) => ({ ...prev, [key]: value }));
  }

  function applyPreset(preset: PredictionInput) {
    setFormValues(preset);
  }

  return (
    <main>
      <div className="navbar">
        <div className="nav-block">
          <small>AI Health Risk System</small>
          <span>Preventive Intelligence Dashboard</span>
        </div>
        <div className="nav-block">
          <small>AI Status</small>
          <span>Online</span>
        </div>
        <div className="nav-block">
          <small>Prediction State</small>
          <span>{predictionState}</span>
        </div>
      </div>

      <section className="hero">
        <div className="section-label">AI-powered preventive healthcare intelligence</div>
        <h1>Personalized Health Risk Scoring</h1>
        <p>
          Predict cardiovascular risk with explainable AI, visualize the top drivers, and
          deliver personalized recommendations in seconds.
        </p>
      </section>

      <div className="layout">
        <aside className="sidebar">
          <div className="glass-card">
            <div className="card-title">Demo Presets</div>
            <div className="demo-buttons">
              <button type="button" onClick={() => applyPreset(defaultValues)}>
                Custom
              </button>
              <button type="button" onClick={() => applyPreset(highRiskPreset)}>
                High Risk
              </button>
              <button type="button" onClick={() => applyPreset(lowRiskPreset)}>
                Low Risk
              </button>
            </div>
          </div>

          <form className="glass-card" onSubmit={handleSubmit}>
            <div className="card-title">Patient Inputs</div>
            <div className="section-label">Demographics</div>
            <div className="input-section">
              <div className="form-group">
                <label htmlFor="age">Age</label>
                <input
                  id="age"
                  type="range"
                  min={18}
                  max={80}
                  value={formValues.age}
                  onChange={(event) => updateField("age", Number(event.target.value))}
                />
                <input
                  type="number"
                  min={18}
                  max={80}
                  value={formValues.age}
                  onChange={(event) => updateField("age", Number(event.target.value))}
                />
              </div>
              <div className="input-divider" />
              <div className="form-group">
                <label htmlFor="gender">Gender</label>
                <select
                  id="gender"
                  value={formValues.gender}
                  onChange={(event) => updateField("gender", event.target.value)}
                >
                  <option>Male</option>
                  <option>Female</option>
                </select>
              </div>
            </div>

            <div className="section-label">Lifestyle</div>
            <div className="input-section">
              <div className="form-group">
                <label htmlFor="smoking">Smoking</label>
                <select
                  id="smoking"
                  value={formValues.smoking}
                  onChange={(event) => updateField("smoking", event.target.value)}
                >
                  <option>Yes</option>
                  <option>No</option>
                </select>
              </div>
              <div className="input-divider" />
              <div className="form-group">
                <label htmlFor="alcohol">Alcohol Consumption</label>
                <select
                  id="alcohol"
                  value={formValues.alcohol_consumption}
                  onChange={(event) =>
                    updateField("alcohol_consumption", event.target.value)
                  }
                >
                  <option>Yes</option>
                  <option>No</option>
                </select>
              </div>
              <div className="input-divider" />
              <div className="form-group">
                <label htmlFor="activity">Physical Activity</label>
                <select
                  id="activity"
                  value={formValues.physical_activity}
                  onChange={(event) =>
                    updateField("physical_activity", event.target.value)
                  }
                >
                  <option>Low</option>
                  <option>Medium</option>
                  <option>High</option>
                </select>
              </div>
              <div className="input-divider" />
              <div className="form-group">
                <label htmlFor="family">Family History</label>
                <select
                  id="family"
                  value={formValues.family_history}
                  onChange={(event) => updateField("family_history", event.target.value)}
                >
                  <option>Yes</option>
                  <option>No</option>
                </select>
              </div>
            </div>

            <div className="section-label">Clinical Indicators</div>
            <div className="input-section">
              <div className="form-group">
                <label htmlFor="bmi">BMI</label>
                <input
                  id="bmi"
                  type="number"
                  min={15}
                  max={45}
                  step={0.1}
                  value={formValues.bmi}
                  onChange={(event) => updateField("bmi", Number(event.target.value))}
                />
              </div>
              <div className="input-divider" />
              <div className="form-group">
                <label htmlFor="sys">Systolic BP</label>
                <input
                  id="sys"
                  type="range"
                  min={90}
                  max={200}
                  value={formValues.systolic_bp}
                  onChange={(event) =>
                    updateField("systolic_bp", Number(event.target.value))
                  }
                />
                <input
                  type="number"
                  min={90}
                  max={200}
                  value={formValues.systolic_bp}
                  onChange={(event) =>
                    updateField("systolic_bp", Number(event.target.value))
                  }
                />
              </div>
              <div className="input-divider" />
              <div className="form-group">
                <label htmlFor="dia">Diastolic BP</label>
                <input
                  id="dia"
                  type="range"
                  min={60}
                  max={130}
                  value={formValues.diastolic_bp}
                  onChange={(event) =>
                    updateField("diastolic_bp", Number(event.target.value))
                  }
                />
                <input
                  type="number"
                  min={60}
                  max={130}
                  value={formValues.diastolic_bp}
                  onChange={(event) =>
                    updateField("diastolic_bp", Number(event.target.value))
                  }
                />
              </div>
              <div className="input-divider" />
              <div className="form-group">
                <label htmlFor="cholesterol">Cholesterol (mg/dL)</label>
                <input
                  id="cholesterol"
                  type="range"
                  min={120}
                  max={320}
                  value={formValues.cholesterol_mg_dl}
                  onChange={(event) =>
                    updateField("cholesterol_mg_dl", Number(event.target.value))
                  }
                />
                <input
                  type="number"
                  min={120}
                  max={320}
                  value={formValues.cholesterol_mg_dl}
                  onChange={(event) =>
                    updateField("cholesterol_mg_dl", Number(event.target.value))
                  }
                />
              </div>
            </div>

            <div className="section-label">Physiological Metrics</div>
            <div className="input-section">
              <div className="form-group">
                <label htmlFor="hr">Heart Rate (BPM)</label>
                <input
                  id="hr"
                  type="range"
                  min={50}
                  max={130}
                  value={formValues.heart_rate_bpm}
                  onChange={(event) =>
                    updateField("heart_rate_bpm", Number(event.target.value))
                  }
                />
                <input
                  type="number"
                  min={50}
                  max={130}
                  value={formValues.heart_rate_bpm}
                  onChange={(event) =>
                    updateField("heart_rate_bpm", Number(event.target.value))
                  }
                />
              </div>
              <div className="input-divider" />
              <div className="form-group">
                <label htmlFor="sdnn">SDNN (HRV)</label>
                <input
                  id="sdnn"
                  type="number"
                  min={10}
                  max={140}
                  step={0.5}
                  value={formValues.sdnn_hrv}
                  onChange={(event) =>
                    updateField("sdnn_hrv", Number(event.target.value))
                  }
                />
              </div>
              <div className="input-divider" />
              <div className="form-group">
                <label htmlFor="rmssd">RMSSD (HRV)</label>
                <input
                  id="rmssd"
                  type="number"
                  min={10}
                  max={140}
                  step={0.5}
                  value={formValues.rmssd_hrv}
                  onChange={(event) =>
                    updateField("rmssd_hrv", Number(event.target.value))
                  }
                />
              </div>
              <div className="input-divider" />
              <div className="form-group">
                <label htmlFor="spo2">SpO2 (%)</label>
                <input
                  id="spo2"
                  type="range"
                  min={90}
                  max={100}
                  step={0.5}
                  value={formValues.spo2}
                  onChange={(event) => updateField("spo2", Number(event.target.value))}
                />
                <input
                  type="number"
                  min={90}
                  max={100}
                  step={0.5}
                  value={formValues.spo2}
                  onChange={(event) => updateField("spo2", Number(event.target.value))}
                />
              </div>
            </div>

            <button className="primary-button" type="submit" disabled={loading}>
              {loading ? "Analyzing..." : "Predict Health Risk"}
            </button>
          </form>
        </aside>

        <section className="content">
          <div className="results-grid">
            <div className="metric-card results-card">
              <div className="card-header">
                <div>
                  <div className="section-label">Risk Score</div>
                  <div className="card-subtitle">Predicted health risk level</div>
                </div>
                <div className="card-subtitle">Score (0-100)</div>
              </div>
              <div className="metric-score">
                {prediction ? prediction.risk_score.toFixed(2) : "--"}
              </div>
              {prediction ? (
                <div className={riskPillClass(prediction.risk_level)}>
                  {prediction.risk_level}
                </div>
              ) : (
                <div className="empty-state">Awaiting prediction.</div>
              )}
              <div className="risk-bar">
                <div className="risk-bar-fill" style={{ width: `${riskPercent}%` }} />
              </div>
            </div>

            <div className="glass-card results-card">
              <div className="card-header">
                <div>
                  <div className="card-title">Personalized Recommendations</div>
                  <div className="card-subtitle">Actionable next steps</div>
                </div>
              </div>
              <div className="recommendation-list">
                {prediction?.recommendations.map((rec) => (
                  <div key={rec} className="insight-card">
                    {rec}
                  </div>
                ))}
                {!prediction && (
                  <div className="empty-state">Recommendations appear after prediction.</div>
                )}
              </div>
            </div>

            <div className="glass-card results-card results-span-2">
              <div className="card-header">
                <div>
                  <div className="card-title">AI Explanation</div>
                  <div className="card-subtitle">Key factors driving the score</div>
                </div>
              </div>
              <div className="ai-explanation-grid">
                <div>
                  <div className="section-label">Top Contributors</div>
                  <div className="chip-grid">
                    {prediction?.positive_contributors.map((item) => (
                      <div key={item.feature} className="chip positive">
                        <span>+</span>
                        {item.feature}
                      </div>
                    ))}
                    {prediction?.negative_contributors.map((item) => (
                      <div key={item.feature} className="chip negative">
                        <span>-</span>
                        {item.feature}
                      </div>
                    ))}
                    {!prediction && <div className="empty-state">Awaiting prediction.</div>}
                  </div>
                </div>
                <div>
                  <div className="section-label">Insights</div>
                  <div className="insight-grid">
                    {prediction?.insights.map((insight) => (
                      <div key={insight} className="insight-card">
                        {insight}
                      </div>
                    ))}
                    {!prediction && <div className="empty-state">No insights yet.</div>}
                  </div>
                </div>
              </div>
            </div>

            <div className="glass-card results-card results-span-2">
              <div className="card-header">
                <div>
                  <div className="card-title">SHAP Visualizations</div>
                  <div className="card-subtitle">Explainability charts</div>
                </div>
              </div>
              {error && <p style={{ color: "var(--danger)" }}>{error}</p>}
              {!prediction && (
                <p style={{ color: "var(--muted)" }}>Run a prediction to load charts.</p>
              )}
              {prediction && (
                <div className="chart-grid">
                  <div className="chart-card">
                    <div className="section-label">Summary Plot</div>
                    <img
                      src={`data:image/png;base64,${prediction.summary_plot}`}
                      alt="SHAP Summary"
                    />
                  </div>
                  <div className="chart-card">
                    <div className="section-label">Bar Plot</div>
                    <img
                      src={`data:image/png;base64,${prediction.bar_plot}`}
                      alt="SHAP Bar"
                    />
                  </div>
                  <div className="chart-card">
                    <div className="section-label">Waterfall Plot</div>
                    <img
                      src={`data:image/png;base64,${prediction.waterfall_plot}`}
                      alt="SHAP Waterfall"
                    />
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>
      </div>

      <footer className="footer">
        <div className="section-label">AI Healthcare System</div>
        <div>Sensor-Augmented Personalized Health Risk Scoring</div>
      </footer>
    </main>
  );
}
