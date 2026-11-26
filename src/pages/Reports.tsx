import { api } from "../lib/api";
import { useEffect, useState } from "react";
import { Card } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { useNavigate } from "react-router-dom";

interface ScreeningRecord {
  id: number;
  modality: string;
  model_name: string;
  anemia_probability: number;
  estimated_hemoglobin: number | null;
  risk_tier: string;
  roi_confidence: number | null;
  created_at?: string;
}

export default function Reports() {
  const [records, setRecords] = useState<ScreeningRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    api
      .get<{ items: ScreeningRecord[] }>("/api/screenings/history")
      .then((data) => {
        setRecords(data.items || []);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : "Could not load history.");
      })
      .finally(() => {
        setLoading(false);
      });
  }, []);

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Reports & History</h1>
          <p className="text-sm text-gray-500">
            Past anemia screening results and estimated hemoglobin trends.
          </p>
        </div>
        <Button onClick={() => navigate("/dashboard")}>Return to dashboard</Button>
      </div>

      {loading && <p className="text-gray-500">Loading screening history...</p>}
      
      {error && (
        <Card className="p-4 border-red-200 bg-red-50 text-red-700 text-sm">
          Failed to load history ({error}). Check that the backend server is running and the route prefix matches main.py.
        </Card>
      )}

      {!loading && !error && records.length === 0 ? (
        <Card className="p-6 text-center space-y-3">
          <p className="text-gray-600">No screening reports recorded yet.</p>
          <Button onClick={() => navigate("/anemia-screening")}>
            Start Anemia Screening
          </Button>
        </Card>
      ) : (
        <div className="space-y-4">
          {records.map((item : ScreeningRecord) => (
            <Card key={item.id} className="p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold uppercase tracking-wide text-gray-500">
                    {item.modality} Screening
                  </span>
                  {item.created_at && (
                    <span className="text-xs text-gray-400">• {item.created_at}</span>
                  )}
                </div>
                {item.estimated_hemoglobin !== null && (
                  <p className="font-medium text-gray-900">
                    Est. Hemoglobin: <span className="font-semibold">{item.estimated_hemoglobin} g/dL</span>
                  </p>
                )}
                <p className="text-xs text-gray-500">
                  Probability: {(item.anemia_probability * 100).toFixed(1)}%
                  {item.roi_confidence !== null && ` • Confidence: ${(item.roi_confidence * 100).toFixed(0)}%`}
                </p>
              </div>

              <div className="flex items-center gap-3">
                <span className="px-3 py-1 text-sm font-medium rounded-full bg-gray-100 text-gray-800 capitalize">
                  {item.risk_tier} Risk
                </span>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}