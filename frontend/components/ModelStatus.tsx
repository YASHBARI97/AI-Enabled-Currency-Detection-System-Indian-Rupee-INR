import type { HealthResponse } from "@/lib/types";

interface Props {
  health: HealthResponse | null;
  loaded: boolean;
  onRetry: () => void;
}

export default function ModelStatus({ health, loaded, onRetry }: Props) {
  if (!loaded) return <span className="pill">Checking service…</span>;
  if (!health || health.status !== "ok") {
    return (
      <button className="pill bad" onClick={onRetry} title="Retry connection">
        ● Service offline · retry
      </button>
    );
  }
  return (
    <span className="pill good" title={`Abstention threshold ${health.threshold ?? "–"}`}>
      ● Model v{health.model_version} · threshold {health.threshold?.toFixed(2)}
    </span>
  );
}
