import type { PredictionResponse } from "@/lib/types";
import { formatPercent, prettyLabel } from "@/lib/validate";

interface Props {
  loading: boolean;
  result: PredictionResponse | null;
  threshold: number | null;
}

const ERROR_TITLES: Record<string, string> = {
  empty_input: "Empty file",
  unsupported_format: "Unsupported file type",
  unreadable_image: "Unreadable image",
  file_too_large: "File too large",
  inference_error: "Prediction failed",
  model_unavailable: "Service unavailable",
  network_error: "Connection problem",
  client_validation_error: "Cannot use this file",
};

export default function ResultCard({ loading, result, threshold }: Props) {
  if (loading) {
    return (
      <div className="result skeleton" aria-busy="true">
        <div className="spinner" aria-hidden />
        <p>Analysing image…</p>
      </div>
    );
  }
  if (!result) {
    return <p className="muted">Upload an image to see the predicted denomination and confidence.</p>;
  }

  if (result.status === "prediction_available") {
    return (
      <div className="result ok">
        <div className="big-label">{prettyLabel(result.predicted_denomination)}</div>
        <div className="muted">{result.predicted_denomination}</div>
        <ConfidenceBar value={result.confidence ?? 0} threshold={threshold} />
        <Details result={result} />
      </div>
    );
  }

  if (result.status === "low_confidence") {
    return (
      <div className="result warn">
        <div className="big-label small-label">{result.message ?? "Unable to classify confidently"}</div>
        <p>
          Best guess confidence was {formatPercent(result.confidence)}, below the required threshold
          {threshold != null ? ` (${(threshold * 100).toFixed(0)}%)` : ""}. Retake the photo on a plain background in
          good light with the whole note in frame, or verify manually.
        </p>
        <ConfidenceBar value={result.confidence ?? 0} threshold={threshold} />
        <Details result={result} />
      </div>
    );
  }

  return (
    <div className="result error" role="alert">
      <div className="big-label small-label">{ERROR_TITLES[result.status] ?? "Error"}</div>
      <p>{result.message ?? "Something went wrong."}</p>
    </div>
  );
}

function ConfidenceBar({ value, threshold }: { value: number; threshold: number | null }) {
  return (
    <div className="bar-wrap" aria-label={`Confidence ${formatPercent(value)}`}>
      <div className="bar-head">
        <span>Confidence</span>
        <strong>{formatPercent(value)}</strong>
      </div>
      <div className="bar" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(value * 100)}>
        <div className="bar-fill" style={{ width: `${Math.min(100, value * 100)}%` }} />
        {threshold != null && <div className="bar-mark" style={{ left: `${threshold * 100}%` }} title="Abstention threshold" />}
      </div>
    </div>
  );
}

function Details({ result }: { result: PredictionResponse }) {
  return (
    <details className="details">
      <summary>Details</summary>
      {result.top_k && (
        <table className="topk">
          <thead>
            <tr><th>Class</th><th>Score</th></tr>
          </thead>
          <tbody>
            {result.top_k.map((t) => (
              <tr key={t.label}><td>{prettyLabel(t.label)}</td><td>{formatPercent(t.confidence)}</td></tr>
            ))}
          </tbody>
        </table>
      )}
      <p className="muted small">
        Inference: {result.inference_ms ?? "–"} ms · model v{result.model_version ?? "–"}
      </p>
      <pre className="json">{JSON.stringify(result, null, 2)}</pre>
    </details>
  );
}
