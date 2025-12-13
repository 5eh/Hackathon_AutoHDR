"use client";

import { useState, useRef } from "react";
import * as ort from "onnxruntime-web";

export default function Machine() {
  const [result, setResult] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [imagePreview, setImagePreview] = useState<string>("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const preprocessImage = (canvas: HTMLCanvasElement): Float32Array => {
    const ctx = canvas.getContext("2d")!;
    const imageData = ctx.getImageData(0, 0, 224, 224);
    const { data } = imageData;

    const float32Data = new Float32Array(3 * 224 * 224);
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    for (let i = 0; i < 224 * 224; i++) {
      float32Data[i] = (data[i * 4] / 255.0 - mean[0]) / std[0];
      float32Data[i + 224 * 224] = (data[i * 4 + 1] / 255.0 - mean[1]) / std[1];
      float32Data[i + 2 * 224 * 224] =
        (data[i * 4 + 2] / 255.0 - mean[2]) / std[2];
    }

    return float32Data;
  };

  const getClassName = async (classIndex: number): Promise<string> => {
    try {
      const response = await fetch(
        "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json",
      );
      const labels = await response.json();
      return labels[classIndex] || `Unknown class ${classIndex}`;
    } catch (error) {
      return `Class ${classIndex}`;
    }
  };

  const analyzeImage = async (file: File) => {
    try {
      setLoading(true);
      setResult("");

      const imageUrl = URL.createObjectURL(file);
      setImagePreview(imageUrl);

      const img = new Image();
      await new Promise((resolve) => {
        img.onload = resolve;
        img.src = imageUrl;
      });

      const canvas = document.createElement("canvas");
      canvas.width = 224;
      canvas.height = 224;
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(img, 0, 0, 224, 224);

      const inputData = preprocessImage(canvas);
      const inputTensor = new ort.Tensor(
        "float32",
        inputData,
        [1, 3, 224, 224],
      );

      const session = await ort.InferenceSession.create("/resnet50.onnx");
      const feeds = { data: inputTensor };
      const output = await session.run(feeds);

      const outputTensor = output[Object.keys(output)[0]];
      const predictions = Array.from(outputTensor.data as Float32Array);

      const topPredictions = await Promise.all(
        predictions
          .map((score, index) => ({ score, index }))
          .sort((a, b) => b.score - a.score)
          .slice(0, 5)
          .map(async (p) => ({
            class: await getClassName(p.index),
            confidence: `${(Math.exp(p.score) * 100).toFixed(1)}%`,
            index: p.index,
          })),
      );

      setResult(JSON.stringify(topPredictions, null, 2));
      URL.revokeObjectURL(imageUrl);
    } catch (error) {
      console.error("Analysis Error:", error);
      setResult(`Error: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith("image/")) {
      analyzeImage(file);
    }
  };

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-4">
        Machine Learning Image Classifier
      </h1>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        className="mb-4"
      />

      {imagePreview && (
        <img src={imagePreview} alt="Preview" className="max-w-xs mb-4" />
      )}

      {loading && <p>Analyzing image...</p>}

      {result && (
        <div className="mt-4 p-4  rounded">
          <h3 className="font-semibold">Classification Results:</h3>
          <pre className="text-sm overflow-auto">{result}</pre>
        </div>
      )}
    </div>
  );
}
