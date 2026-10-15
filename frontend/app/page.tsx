"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Dropzone from "@/components/Dropzone";
import ModelStatus from "@/components/ModelStatus";
import ResultCard from "@/components/ResultCard";
import HistoryList from "@/components/HistoryList";
import { fetchHealth, predictImage } from "@/lib/api";
import { checkFile } from "@/lib/validate";
import type { HealthResponse, HistoryItem, PredictionResponse } from "@/lib/types";

const DISCLAIMER =
  "Denomination estimate only. This does NOT verify authenticity or legal tender status. " +
  "Poor lighting, wear, folds, cropping or unsupported note designs can cause errors. " +
  "Verify manually for any consequential decision.";

export default function Home() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoaded, setHealthLoaded] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const abortRef = useRef<AbortController | null>(null);
  const idRef = useRef(0);

  const refreshHealth = useCallback(async () => {
    setHealth(await fetchHealth());
    setHealthLoaded(true);
  }, []);

  useEffect(() => {
    refreshHealth();
  }, [refreshHealth]);

  // Release object URLs to avoid leaks.
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  const handleFile = useCallback(
    async (file: File) => {
      abortRef.current?.abort();
      const maxMb = health?.max_upload_mb ?? 5;
      const check = checkFile(file, maxMb);
      setResult(null);
      setFileName(file.name);
      if (!check.ok) {
        setPreview(null);
        setResult({
          status: "client_validation_error",
          predicted_denomination: null,
          confidence: null,
          inference_ms: null,
          message: check.message,
        });
        return;
      }
      setPreview(URL.createObjectURL(file));
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      setLoading(true);
      try {
        const res = await predictImage(file, ctrl.signal);
        setResult(res);
        if (res.status === "model_unavailable") refreshHealth();
        setHistory((h) =>
          [{ id: ++idRef.current, time: new Date().toLocaleTimeString(), fileName: file.name, result: res }, ...h].slice(0, 8),
        );
      } catch {
        /* aborted by a newer upload */
      } finally {
        if (abortRef.current === ctrl) setLoading(false);
      }
    },
    [health, refreshHealth],
  );

  const reset = () => {
    abortRef.current?.abort();
    setPreview(null);
    setResult(null);
    setFileName("");
    setLoading(false);
  };

  const modelDown = healthLoaded && (!health || health.status !== "ok");

  return (
    <main className="container">
      <header className="header">
        <div>
          <h1>AI Enabled Currency Detection</h1>
          <p className="muted">Indian Rupee (INR) denomination classifier · educational prototype</p>
        </div>
        <ModelStatus health={health} loaded={healthLoaded} onRetry={refreshHealth} />
      </header>

      <div className="notice" role="note">
        <strong>Important:</strong> {DISCLAIMER}
      </div>

      {modelDown && (
        <div className="alert error" role="alert">
          The prediction service is unavailable. Start the backend (<code>uvicorn app.api:app</code>) and make sure a
          trained model is in <code>artifacts/</code>, then retry.
        </div>
      )}

      <section className="grid">
        <div className="panel">
          <h2>1. Upload a single banknote</h2>
          <Dropzone onFile={handleFile} disabled={loading || modelDown} maxMb={health?.max_upload_mb ?? 5} />
          {preview && (
            <figure className="preview">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={preview} alt={`Uploaded banknote: ${fileName}`} />
              <figcaption className="muted">{fileName}</figcaption>
            </figure>
          )}
          {(preview || result) && (
            <button className="btn secondary" onClick={reset} disabled={loading}>
              Clear
            </button>
          )}
          {health?.classes?.length ? (
            <p className="muted small">Supported: {health.classes.join(", ")}</p>
          ) : null}
        </div>

        <div className="panel" aria-live="polite">
          <h2>2. Result</h2>
          <ResultCard loading={loading} result={result} threshold={health?.threshold ?? null} />
        </div>
      </section>

      <HistoryList items={history} />

      <footer className="footer muted small">
        Uploaded images are processed in memory and are not stored. Confidence is a model score, not a calibrated
        probability.
      </footer>
    </main>
  );
}
