"use client";

import React, { useState, useRef, useCallback } from "react";
import * as ort from "onnxruntime-web";

const RealEstateEnhancer: React.FC = () => {
  const [image, setImage] = useState<string | null>(null);
  const [enhancedImage, setEnhancedImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  const [error, setError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const initializeModel = useCallback(async () => {
    try {
      if (session) return session;

      setIsLoading(true);
      setError(null);

      // Configure ONNX Runtime
      ort.env.wasm.wasmPaths =
        "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/";

      // Try different execution providers
      const providers: ort.ExecutionProvider[] = ["webgl", "wasm"];

      const newSession = await ort.InferenceSession.create("/model.onnx", {
        executionProviders: providers,
      });

      setSession(newSession);
      console.log("✅ Model loaded successfully!");
      return newSession;
    } catch (err) {
      const errorMsg = `Failed to load model: ${err instanceof Error ? err.message : "Unknown error"}`;
      setError(errorMsg);
      console.error("❌ Model loading error:", err);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [session]);

  // Preprocess image for model input
  const preprocessImage = (imageElement: HTMLImageElement): Float32Array => {
    const canvas = canvasRef.current;
    if (!canvas) throw new Error("Canvas not available");

    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Could not get canvas context");

    // Resize to model input size
    canvas.width = 512;
    canvas.height = 512;
    ctx.drawImage(imageElement, 0, 0, 512, 512);

    // Get image data
    const imageData = ctx.getImageData(0, 0, 512, 512);
    const { data } = imageData;

    // Convert to model input format [1, 3, 512, 512]
    const input = new Float32Array(1 * 3 * 512 * 512);

    for (let i = 0; i < 512 * 512; i++) {
      const pixelIndex = i * 4;
      // Normalize RGB values from [0, 255] to [-1, 1]
      input[i] = (data[pixelIndex] / 255.0) * 2.0 - 1.0; // R
      input[i + 512 * 512] = (data[pixelIndex + 1] / 255.0) * 2.0 - 1.0; // G
      input[i + 512 * 512 * 2] = (data[pixelIndex + 2] / 255.0) * 2.0 - 1.0; // B
    }

    return input;
  };

  // Postprocess model output to image
  const postprocessOutput = (output: Float32Array): string => {
    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Could not create canvas context");

    const imageData = ctx.createImageData(512, 512);
    const { data } = imageData;

    // Convert from model output format back to RGBA
    for (let i = 0; i < 512 * 512; i++) {
      const pixelIndex = i * 4;
      // Denormalize from [-1, 1] to [0, 255]
      data[pixelIndex] = Math.max(0, Math.min(255, (output[i] + 1.0) * 127.5)); // R
      data[pixelIndex + 1] = Math.max(
        0,
        Math.min(255, (output[i + 512 * 512] + 1.0) * 127.5),
      ); // G
      data[pixelIndex + 2] = Math.max(
        0,
        Math.min(255, (output[i + 512 * 512 * 2] + 1.0) * 127.5),
      ); // B
      data[pixelIndex + 3] = 255; // Alpha
    }

    ctx.putImageData(imageData, 0, 0);
    return canvas.toDataURL("image/jpeg", 0.95);
  };

  // Handle image upload
  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setError(null);
    setEnhancedImage(null);

    const reader = new FileReader();
    reader.onload = (e) => {
      setImage(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  };

  // Enhance image with AI
  const enhanceImage = async () => {
    if (!image) return;

    try {
      setIsLoading(true);
      setError(null);

      // Initialize model if needed
      const modelSession = await initializeModel();

      // Create image element
      const img = new Image();
      img.crossOrigin = "anonymous";

      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = () => reject(new Error("Failed to load image"));
        img.src = image;
      });

      // Preprocess image
      const inputTensor = preprocessImage(img);
      const tensorData = new ort.Tensor(
        "float32",
        inputTensor,
        [1, 3, 512, 512],
      );

      // Run inference
      console.log("🔮 Running AI enhancement...");
      const results = await modelSession.run({ input: tensorData });

      // Get output tensor
      const outputTensor = results.output;
      if (!outputTensor) throw new Error("No output from model");

      // Postprocess and display result
      const enhancedDataURL = postprocessOutput(
        outputTensor.data as Float32Array,
      );
      setEnhancedImage(enhancedDataURL);

      console.log("✅ Enhancement complete!");
    } catch (err) {
      const errorMsg = `Enhancement failed: ${err instanceof Error ? err.message : "Unknown error"}`;
      setError(errorMsg);
      console.error("❌ Enhancement error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  // Download enhanced image
  const downloadImage = () => {
    if (!enhancedImage) return;

    const link = document.createElement("a");
    link.download = "enhanced-real-estate.jpg";
    link.href = enhancedImage;
    link.click();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-purple-900 to-indigo-900 py-12 px-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-white mb-4">
            🏠 Real Estate AI Enhancer
          </h1>
          <p className="text-xl text-blue-200">
            Transform your property photos with AI-powered enhancement
          </p>
        </div>

        {/* Upload Section */}
        <div className="bg-white/10 backdrop-blur-md rounded-2xl p-8 mb-8 border border-white/20">
          <div className="text-center">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700
                       text-white px-8 py-4 rounded-xl font-semibold text-lg transition-all duration-300
                       transform hover:scale-105 shadow-lg hover:shadow-xl"
            >
              📷 Upload Real Estate Photo
            </button>
            <p className="text-blue-200 mt-4">
              Supports JPG, PNG, WebP formats
            </p>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-500/20 border border-red-500/50 rounded-xl p-4 mb-8 text-red-200">
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* Image Display */}
        {image && (
          <div className="grid md:grid-cols-2 gap-8 mb-8">
            {/* Original Image */}
            <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20">
              <h3 className="text-xl font-semibold text-white mb-4">
                Original Photo
              </h3>
              <img
                src={image}
                alt="Original"
                className="w-full h-64 object-cover rounded-xl shadow-lg"
              />
            </div>

            {/* Enhanced Image */}
            <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20">
              <h3 className="text-xl font-semibold text-white mb-4">
                AI Enhanced
              </h3>
              {enhancedImage ? (
                <div>
                  <img
                    src={enhancedImage}
                    alt="Enhanced"
                    className="w-full h-64 object-cover rounded-xl shadow-lg mb-4"
                  />
                  <button
                    onClick={downloadImage}
                    className="bg-green-600 hover:bg-green-700 text-white px-6 py-2 rounded-lg
                             font-semibold transition-all duration-300 transform hover:scale-105"
                  >
                    💾 Download Enhanced
                  </button>
                </div>
              ) : (
                <div className="w-full h-64 bg-gray-800/50 rounded-xl flex items-center justify-center">
                  <span className="text-gray-400">
                    Enhanced image will appear here
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Action Buttons */}
        {image && (
          <div className="text-center">
            <button
              onClick={enhanceImage}
              disabled={isLoading}
              className="bg-gradient-to-r from-green-500 to-blue-600 hover:from-green-600 hover:to-blue-700
                       disabled:from-gray-500 disabled:to-gray-600 disabled:cursor-not-allowed
                       text-white px-12 py-4 rounded-xl font-semibold text-xl transition-all duration-300
                       transform hover:scale-105 shadow-lg hover:shadow-xl"
            >
              {isLoading ? "🔮 Enhancing..." : "✨ Enhance with AI"}
            </button>
          </div>
        )}

        {/* Canvas for image processing */}
        <canvas ref={canvasRef} className="hidden" />

        {/* Tech Info */}
        <div className="mt-16 text-center text-blue-200 space-y-2">
          <p>
            🔒 <strong>100% Private:</strong> All processing happens in your
            browser
          </p>
          <p>
            ⚡ <strong>Fast:</strong> Client-side AI with WebGL acceleration
          </p>
          <p>
            🌍 <strong>No Upload:</strong> Your images never leave your device
          </p>
        </div>
      </div>
    </div>
  );
};

export default RealEstateEnhancer;
