"use client";

import React, { useRef, useState, useEffect, useCallback } from "react";
import * as ort from "onnxruntime-web";

// --- CONFIGURATION ---
interface ModelInput {
  name: string;
  dims: [1, 3, number, number];
}

const MODEL_PATH = "/ai/model.onnx";
const INPUT_META: ModelInput = {
  name: "input",
  dims: [1, 3, 256, 256],
};
const OUTPUT_NAME = "output";

const ModelViewer: React.FC = () => {
  const inputCanvasRef = useRef<HTMLCanvasElement>(null);
  const outputCanvasRef = useRef<HTMLCanvasElement>(null);
  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [modelLoadedMessage, setModelLoadedMessage] = useState("");
  const [inferenceResult, setInferenceResult] = useState<string>(
    "No inference run. Awaiting image upload.",
  );

  // --- MODEL LOADING (Optimized Threads) ---
  useEffect(() => {
    ort.env.wasm.numThreads = 4; // Multi-threading enabled
    ort.env.wasm.simd = true;
    ort.env.wasm.proxy = true;
    ort.env.logLevel = "verbose";

    console.log("ORT: Setting up WASM environment (4 threads enabled)...");

    const loadModel = async () => {
      try {
        console.log(`ORT: Attempting to load model from ${MODEL_PATH}...`);
        const newSession = await ort.InferenceSession.create(MODEL_PATH, {
          executionProviders: ["wasm"],
        });
        setSession(newSession);
        setIsLoading(false);
        setModelLoadedMessage(
          `✅ Model loaded successfully. Input: ${newSession.inputNames.join(", ")}, Output: ${newSession.outputNames.join(", ")}`,
        );
        console.log("ORT: InferenceSession created:", newSession);
      } catch (e) {
        console.error("ORT: Failed to load/create InferenceSession.", e);
        setModelLoadedMessage(
          `❌ Failed to load model: ${e instanceof Error ? e.message : String(e)}`,
        );
        setIsLoading(false);
      }
    };

    loadModel();
  }, []);

  // --- PREPROCESSING (RETAINS ImageNet for safety, as input was fine) ---
  const preprocessImage = (imageData: ImageData): Float32Array => {
    const [_, C, H, W] = INPUT_META.dims;
    const tensorData = new Float32Array(C * H * W);
    const { data, width } = imageData;

    // Use common ImageNet normalization constants for input
    const [meanR, meanG, meanB] = [0.485, 0.456, 0.406];
    const [stdR, stdG, stdB] = [0.229, 0.224, 0.225];

    let k = 0;
    for (let i = 0; i < H; i++) {
      for (let j = 0; j < W; j++) {
        const pixelIndex = (i * width + j) * 4;

        const r = data[pixelIndex + 0] / 255;
        const g = data[pixelIndex + 1] / 255;
        const b = data[pixelIndex + 2] / 255;

        // Channel-wise, then row-wise (CHW)
        tensorData[k] = (r - meanR) / stdR;
        tensorData[k + H * W] = (g - meanG) / stdR;
        tensorData[k + H * W * 2] = (b - meanB) / stdB;

        k++;
      }
    }

    console.log(
      "Preprocessing: Image successfully converted to Float32Array tensor data (CHW format).",
    );
    return tensorData;
  };

  // --- POST-PROCESSING (Creative Diagnostic Applied) ---
  const postProcessAndRender = (outputTensor: ort.Tensor) => {
    const outputData = outputTensor.data as Float32Array;
    const [_, C, H, W] = outputTensor.dims as [number, number, number, number];

    const canvas = outputCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = W;
    canvas.height = H;

    const imageData = ctx.createImageData(W, H);
    const outputPixels = imageData.data;

    // --- FIX: SCENARIO A - Assume output is scaled [0, 1] ---
    console.log(
      "Post-processing: Attempting simple [0, 1] scaling and BGR-to-RGB channel swap.",
    );

    let k = 0;
    for (let i = 0; i < H; i++) {
      for (let j = 0; j < W; j++) {
        const pixelIndex = (i * W + j) * 4;

        // CHW Indices
        const r_chw_index = k;
        const g_chw_index = k + H * W;
        const b_chw_index = k + H * W * 2;

        // Creative Fix: Simple [0, 1] scaling to 0-255
        const r = Math.min(255, Math.max(0, outputData[r_chw_index] * 255));
        const g = Math.min(255, Math.max(0, outputData[g_chw_index] * 255));
        const b = Math.min(255, Math.max(0, outputData[b_chw_index] * 255));

        // BGR -> RGB Channel Swap (retained as it's common)
        outputPixels[pixelIndex + 0] = b; // R slot gets Blue data (b)
        outputPixels[pixelIndex + 1] = g; // G slot gets Green data (g)
        outputPixels[pixelIndex + 2] = r; // B slot gets Red data (r)
        outputPixels[pixelIndex + 3] = 255; // Alpha

        k++;
      }
    }

    ctx.putImageData(imageData, 0, 0);

    console.log(
      `Post-processing: Output image (${W}x${H}) successfully rendered to the output canvas.`,
    );
    return `Output image (Type: Image-to-Image) successfully rendered with simplified [0, 1] de-normalization.`;
  };

  // --- MAIN INFERENCE FUNCTION ---
  const runInference = useCallback(
    async (tensorData: Float32Array) => {
      if (!session) {
        setInferenceResult("Error: Inference session is not loaded.");
        return;
      }

      try {
        setInferenceResult("Running inference...");
        console.log("ORT: Creating input tensor...");

        const [N, C, H, W] = INPUT_META.dims;
        const inputTensor = new ort.Tensor("float32", tensorData, [N, C, H, W]);

        // FIX: Using the generic Record<string, ort.Tensor> type to avoid TS 2694 errors
        const feeds: Record<string, ort.Tensor> = {
          [INPUT_META.name]: inputTensor,
        };

        const start = performance.now();
        const results = await session.run(feeds);
        const end = performance.now();
        const inferenceTime = (end - start).toFixed(2);

        console.log(
          `ORT: session.run() complete. Inference time: ${inferenceTime}ms.`,
        );

        const outputTensor = results[OUTPUT_NAME];
        if (!outputTensor) {
          throw new Error(`Output tensor named '${OUTPUT_NAME}' not found.`);
        }

        console.log("ORT: Output Tensor Details:", {
          dims: outputTensor.dims,
          type: outputTensor.type,
          size: outputTensor.size,
        });

        const processedMessage = postProcessAndRender(outputTensor);
        setInferenceResult(
          `Inference Time: ${inferenceTime}ms. | ${processedMessage}`,
        );
      } catch (e) {
        console.error("ORT: Error during inference.", e);
        setInferenceResult(
          `Inference Error: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    },
    [session],
  );

  // --- IMAGE FILE HANDLER ---
  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        const inputCanvas = inputCanvasRef.current;
        if (!inputCanvas) return;
        const ctx = inputCanvas.getContext("2d");
        if (!ctx) return;

        const [_, __, H, W] = INPUT_META.dims;
        inputCanvas.width = W;
        inputCanvas.height = H;

        // Draw and resize the uploaded image to the model's required input dimensions
        ctx.drawImage(img, 0, 0, W, H);

        const imageData = ctx.getImageData(0, 0, W, H);
        const tensorData = preprocessImage(imageData);
        runInference(tensorData);
      };
      img.src = e.target?.result as string;
    };
    reader.readAsDataURL(file);
  };

  // --- JSX RENDER ---
  return (
    <div style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
      <h1>💡 ONNX Runtime Web Model Viewer (Image-to-Image)</h1>
      <p>Model Status: **{isLoading ? "Loading..." : modelLoadedMessage}**</p>
      <hr />

      <div style={{ display: "flex", gap: "40px" }}>
        {/* Input Section */}
        <div style={{ flex: 1 }}>
          <h3>1. Image Input (Input Canvas)</h3>
          <input
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            disabled={isLoading || !session}
          />
          <div
            style={{
              marginTop: "10px",
              border: "1px solid #ccc",
              padding: "5px",
              backgroundColor: "#f0f0f0",
            }}
          >
            <canvas
              ref={inputCanvasRef}
              style={{ maxWidth: "100%", height: "auto" }}
            />
            <p style={{ fontSize: "0.8em", textAlign: "center" }}>
              **Input Image ({INPUT_META.dims[3]}x{INPUT_META.dims[2]})**
            </p>
          </div>
        </div>

        {/* Output Section */}
        <div style={{ flex: 1 }}>
          <h3>2. Model Output (Output Canvas)</h3>
          <div
            style={{
              marginTop: "10px",
              border: "1px solid #4CAF50",
              padding: "5px",
              backgroundColor: "#e8f5e9",
            }}
          >
            <canvas
              ref={outputCanvasRef}
              style={{ maxWidth: "100%", height: "auto" }}
            />
            <p style={{ fontSize: "0.8em", textAlign: "center" }}>
              **Inference Output Visualization ({INPUT_META.dims[3]}x
              {INPUT_META.dims[2]})**
            </p>
          </div>
        </div>
      </div>

      <h3 style={{ marginTop: "20px" }}>3. Inference Log</h3>
      <div
        style={{
          border: "1px solid #007BFF",
          padding: "15px",
          backgroundColor: "#e6f7ff",
          wordBreak: "break-all",
        }}
      >
        <pre>{inferenceResult}</pre>
        <p style={{ marginTop: "10px", fontSize: "0.9em" }}>
          *Check your browser **Developer Console** for FULL ONNX Runtime and
          detailed process logs.*
        </p>
      </div>
    </div>
  );
};

export default ModelViewer;
