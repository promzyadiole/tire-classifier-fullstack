"use client";

import { useRef, useState } from "react";
import { classifyTireImage, TirePredictionResponse } from "@/lib/api";

export default function TireClassifier() {
  const inputRef = useRef<HTMLInputElement | null>(null);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<TirePredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];

    if (!file) return;

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError("");
  }

  async function handleSubmit() {
    if (!selectedFile) {
      setError("Please upload a tire image first.");
      return;
    }

    try {
      setLoading(true);
      setError("");
      setResult(null);

      const prediction = await classifyTireImage(selectedFile);
      setResult(prediction);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Something went wrong.");
      }
    } finally {
      setLoading(false);
    }
  }

  function handleClear() {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError("");

    if (inputRef.current) {
      inputRef.current.value = "";
    }
  }

  return (
    <div className="mx-auto grid max-w-6xl gap-8 lg:grid-cols-2">
      <div className="rounded-3xl border border-zinc-200 bg-white p-6 shadow-sm">
        <div className="mb-5">
          <h2 className="text-xl font-semibold text-zinc-900">
            Upload Tire Image
          </h2>
          <p className="mt-1 text-sm text-zinc-500">
            Upload a tire image and let the model classify whether it is safe or
            damaged.
          </p>
        </div>

        <div
          onClick={() => inputRef.current?.click()}
          className="flex min-h-[360px] cursor-pointer items-center justify-center rounded-2xl border-2 border-dashed border-zinc-300 bg-zinc-50 transition hover:bg-zinc-100"
        >
          {previewUrl ? (
            <img
              src={previewUrl}
              alt="Uploaded tire preview"
              className="h-full max-h-[420px] w-full rounded-2xl object-cover"
            />
          ) : (
            <div className="text-center">
              <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-zinc-200 text-2xl">
                🛞
              </div>
              <p className="text-sm font-medium text-zinc-700">
                Click to upload tire image
              </p>
              <p className="mt-1 text-xs text-zinc-500">
                PNG, JPG, JPEG supported
              </p>
            </div>
          )}
        </div>

        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="hidden"
        />

        <div className="mt-6 grid grid-cols-2 gap-4">
          <button
            onClick={handleClear}
            disabled={loading}
            className="rounded-xl bg-zinc-100 px-5 py-3 text-sm font-semibold text-zinc-800 transition hover:bg-zinc-200 disabled:opacity-60"
          >
            Clear
          </button>

          <button
            onClick={handleSubmit}
            disabled={loading}
            className="rounded-xl bg-orange-500 px-5 py-3 text-sm font-semibold text-white transition hover:bg-orange-600 disabled:opacity-60"
          >
            {loading ? "Analyzing..." : "Submit"}
          </button>
        </div>
      </div>

      <div className="rounded-3xl border border-zinc-200 bg-white p-6 shadow-sm">
        <div className="mb-5">
          <h2 className="text-xl font-semibold text-zinc-900">
            Classification Result
          </h2>
          <p className="mt-1 text-sm text-zinc-500">
            The prediction result from your FastAPI + PyTorch backend will
            appear here.
          </p>
        </div>

        {!result && !error && (
          <div className="flex min-h-[360px] items-center justify-center rounded-2xl bg-zinc-50">
            <p className="text-sm text-zinc-500">
              No prediction yet. Upload an image and click submit.
            </p>
          </div>
        )}

        {error && (
          <div className="rounded-2xl border border-red-200 bg-red-50 p-5 text-sm text-red-700">
            {error}
          </div>
        )}

        {result && (
          <div className="space-y-5">
            <div
              className={`rounded-2xl p-5 ${
                result.status === "safe"
                  ? "bg-green-50 text-green-800"
                  : "bg-red-50 text-red-800"
              }`}
            >
              <p className="text-sm font-medium">Prediction</p>
              <h3 className="mt-2 text-3xl font-bold">{result.label}</h3>
              <p className="mt-2 text-sm">{result.recommendation}</p>
            </div>

            <div className="rounded-2xl bg-zinc-50 p-5">
              <p className="text-sm font-medium text-zinc-600">Confidence</p>
              <p className="mt-2 text-4xl font-bold text-zinc-900">
                {result.confidence}%
              </p>

              <div className="mt-4 h-3 overflow-hidden rounded-full bg-zinc-200">
                <div
                  className={`h-full rounded-full ${
                    result.status === "safe" ? "bg-green-500" : "bg-red-500"
                  }`}
                  style={{
                    width: `${Math.min(Math.max(result.confidence, 0), 100)}%`,
                  }}
                />
              </div>
            </div>

            <div className="rounded-2xl border border-zinc-200 p-5">
              <p className="text-sm font-medium text-zinc-600">Raw Output</p>
              <p className="mt-2 text-sm text-zinc-800">{result.message}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}