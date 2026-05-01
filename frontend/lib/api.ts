export type TirePredictionResponse = {
  confidence: number;
  label: string;
  status: "safe" | "danger";
  recommendation: string;
  message: string;
};

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export async function classifyTireImage(
  file: File
): Promise<TirePredictionResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_URL}/api/predict`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new Error(error?.detail || "Prediction failed.");
  }

  return response.json();
}