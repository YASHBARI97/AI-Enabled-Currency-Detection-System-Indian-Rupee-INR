import type { HistoryItem } from "@/lib/types";
import { formatPercent, prettyLabel } from "@/lib/validate";

// Session-only history kept in React state: nothing is persisted and images are never retained.
export default function HistoryList({ items }: { items: HistoryItem[] }) {
  if (!items.length) return null;
  return (
    <section className="panel history">
      <h2>This session</h2>
      <div className="table-scroll">
        <table>
          <thead>
            <tr><th>Time</th><th>File</th><th>Outcome</th><th>Confidence</th><th>ms</th></tr>
          </thead>
          <tbody>
            {items.map((h) => (
              <tr key={h.id}>
                <td>{h.time}</td>
                <td className="ellipsis" title={h.fileName}>{h.fileName}</td>
                <td>
                  {h.result.status === "prediction_available"
                    ? prettyLabel(h.result.predicted_denomination)
                    : h.result.status === "low_confidence"
                    ? "Uncertain"
                    : h.result.status.replace(/_/g, " ")}
                </td>
                <td>{formatPercent(h.result.confidence)}</td>
                <td>{h.result.inference_ms ?? "–"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
