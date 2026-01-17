import { useState } from "react";
import { predictHeadline } from "./api";
import "./App.css";

const LABEL_STYLES = {
  fake: { color: "#f97373", bg: "rgba(248, 113, 113, 0.16)" },
  misleading: { color: "#fbbf24", bg: "rgba(251, 191, 36, 0.16)" },
  satire: { color: "#a855f7", bg: "rgba(168, 85, 247, 0.16)" },
  real: { color: "#34d399", bg: "rgba(52, 211, 153, 0.16)" },
};

const LABEL_ORDER = ["fake", "misleading", "satire", "real"];

function App() {
  const [headline, setHeadline] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setResult(null);

    const text = headline.trim();
    if (!text) {
      setError("Please enter a news headline to analyse.");
      return;
    }

    try {
      setLoading(true);
      const res = await predictHeadline(text);
      setResult(res);
    } catch (err) {
      console.error(err);
      setError("Could not reach the API. Is the backend running on port 8000?");
    } finally {
      setLoading(false);
    }
  };

  const renderProbabilities = () => {
    if (!result?.probabilities) return null;
    const entries = LABEL_ORDER.map((key) => [
      key,
      result.probabilities[key],
    ]);

    return (
      <div className="probabilities">
        <h3>Class probabilities</h3>
        <p className="prob-subtitle">
          Shows how confident the model is for each class.
        </p>
        {entries.map(([label, prob]) => {
          const pct = (prob * 100).toFixed(1);
          return (
            <div key={label} className="prob-row">
              <span className="prob-label">{label.toUpperCase()}</span>
              <div className="prob-bar">
                <div
                  className="prob-bar-fill"
                  style={{
                    width: `${pct}%`,
                    backgroundColor: LABEL_STYLES[label].color,
                  }}
                />
              </div>
              <span className="prob-value">{pct}%</span>
            </div>
          );
        })}
      </div>
    );
  };

  const predictedLabel = result?.label_name;
  const labelStyle =
    predictedLabel && LABEL_STYLES[predictedLabel]
      ? LABEL_STYLES[predictedLabel]
      : { color: "#e5e7eb", bg: "rgba(156, 163, 175, 0.16)" };

  const topConfidence =
    result?.probabilities &&
    Math.max(...Object.values(result.probabilities)) * 100;

  return (
    <div className="app-root">
      <div className="glow glow-1" />
      <div className="glow glow-2" />

      <header className="app-header">
        <p className="badge">MSc AI · NLP Project</p>
        <h1>Sri Lanka Misinformation Detection</h1>
        <p className="subtitle">
          Enter a news headline in Sri Lankan context and classify it as{" "}
          <span className="accent">fake</span>,{" "}
          <span className="accent">misleading</span>,{" "}
          <span className="accent">satire</span>, or{" "}
          <span className="accent">real</span>. Powered by ML + NLP.
        </p>

        <div className="label-legend">
          {LABEL_ORDER.map((key) => (
            <div key={key} className="legend-pill">
              <span
                className="legend-dot"
                style={{ backgroundColor: LABEL_STYLES[key].color }}
              />
              <span>{key.toUpperCase()}</span>
            </div>
          ))}
        </div>
      </header>

      <main className="app-main">
        <section className="card card-input">
          <div className="card-header">
            <h2>Try a headline</h2>
            <p>Paste a headline from a Sri Lankan news site or type your own.</p>
          </div>

          <form onSubmit={handleSubmit} className="form">
            <textarea
              rows={3}
              placeholder="e.g. Minister announces plans to strengthen domestic aviation to boost tourism"
              value={headline}
              onChange={(e) => setHeadline(e.target.value)}
            />
            <div className="form-footer">
              <span className="hint">
                Tip: Try clearly fake, political, and satirical headlines to see
                how the model behaves.
              </span>
              <button type="submit" disabled={loading}>
                {loading ? (
                  <span className="btn-spinner">
                    <span className="spinner" /> Analysing…
                  </span>
                ) : (
                  "Classify"
                )}
              </button>
            </div>
          </form>

          {error && <p className="error">{error}</p>}
        </section>

        <section className="card card-result">
          <div className="card-header">
            <h2>Prediction</h2>
            <p>Model output for the most recent headline.</p>
          </div>

          {!result && (
            <div className="empty-state">
              <p>No prediction yet.</p>
              <p className="empty-sub">
                Enter a headline above and click{" "}
                <span className="accent">Classify</span> to see the result.
              </p>
            </div>
          )}

          {result && (
            <>
              <p className="headline-preview">“{result.headline}”</p>

              <div className="prediction-row">
                <span className="prediction-label">Predicted label</span>
                <span
                  className="label-pill"
                  style={{
                    color: labelStyle.color,
                    backgroundColor: labelStyle.bg,
                    borderColor: labelStyle.color,
                  }}
                >
                  {predictedLabel.toUpperCase()}
                </span>
              </div>

              {typeof topConfidence === "number" && (
                <p className="confidence">
                  Confidence:{" "}
                  <span>
                    {topConfidence.toFixed(1)}
                    %
                  </span>
                </p>
              )}

              {renderProbabilities()}
            </>
          )}
        </section>
      </main>

      <footer className="app-footer">
        <span>Built with FastAPI · Scikit-learn · CatBoost · React + Vite</span>
      </footer>
    </div>
  );
}

export default App;
