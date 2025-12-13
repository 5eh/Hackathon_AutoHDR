"use client";

import { useState } from "react";
import * as ort from "onnxruntime-web";

interface BenchmarkResult {
  device: string;
  modelLoadTime: number;
  inferenceTime: number;
  memoryUsage: number;
  cpuCores: number;
  userAgent: string;
  provider: string;
}

export default function Benchmark() {
  const [results, setResults] = useState<BenchmarkResult[]>([]);
  const [running, setRunning] = useState(false);

  const getDeviceInfo = () => {
    const nav = navigator as any;
    return {
      device: nav.platform || "Unknown",
      cpuCores: nav.hardwareConcurrency || 1,
      userAgent: nav.userAgent,
      memory: (nav.deviceMemory || "Unknown") + "GB",
    };
  };

  const benchmarkProvider = async (provider: "cpu" | "webgl" | "wasm") => {
    const deviceInfo = getDeviceInfo();
    const startMemory = (performance as any).memory?.usedJSHeapSize || 0;

    // Model load benchmark
    const loadStart = performance.now();
    const session = await ort.InferenceSession.create("/resnet50.onnx", {
      executionProviders: [provider],
    });
    const loadTime = performance.now() - loadStart;

    // Create dummy input for inference benchmark
    const dummyData = new Float32Array(3 * 224 * 224).fill(0.5);
    const inputTensor = new ort.Tensor("float32", dummyData, [1, 3, 224, 224]);

    // Warmup run
    await session.run({ data: inputTensor });

    // Benchmark inference (average of 5 runs)
    const inferenceStart = performance.now();
    for (let i = 0; i < 5; i++) {
      await session.run({ data: inputTensor });
    }
    const totalInferenceTime = performance.now() - inferenceStart;
    const avgInferenceTime = totalInferenceTime / 5;

    const endMemory = (performance as any).memory?.usedJSHeapSize || 0;
    const memoryUsage = (endMemory - startMemory) / 1024 / 1024; // MB

    return {
      device: deviceInfo.device,
      modelLoadTime: loadTime,
      inferenceTime: avgInferenceTime,
      memoryUsage: memoryUsage,
      cpuCores: deviceInfo.cpuCores,
      userAgent: deviceInfo.userAgent.substring(0, 100),
      provider: provider.toUpperCase(),
    };
  };

  const runBenchmark = async () => {
    setRunning(true);
    setResults([]);

    const providers: ("cpu" | "webgl" | "wasm")[] = ["cpu", "webgl", "wasm"];
    const newResults: BenchmarkResult[] = [];

    for (const provider of providers) {
      try {
        console.log(`Testing ${provider}...`);
        const result = await benchmarkProvider(provider);
        newResults.push(result);
        setResults([...newResults]); // Update UI progressively
      } catch (error) {
        console.error(`${provider} benchmark failed:`, error);
        newResults.push({
          device: getDeviceInfo().device,
          modelLoadTime: -1,
          inferenceTime: -1,
          memoryUsage: -1,
          cpuCores: getDeviceInfo().cpuCores,
          userAgent: getDeviceInfo().userAgent.substring(0, 100),
          provider: `${provider.toUpperCase()} (FAILED)`,
        });
      }
    }

    setRunning(false);
  };

  const exportResults = () => {
    const data = {
      timestamp: new Date().toISOString(),
      results: results,
      deviceInfo: getDeviceInfo(),
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `ai-benchmark-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold mb-6">AI Model Device Benchmark</h1>

      <div className="mb-6">
        <button
          onClick={runBenchmark}
          disabled={running}
          className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 disabled:opacity-50 mr-4"
        >
          {running ? "Running Benchmark..." : "Start Benchmark"}
        </button>

        {results.length > 0 && (
          <button
            onClick={exportResults}
            className="bg-green-600 text-white px-6 py-3 rounded-lg hover:bg-green-700"
          >
            Export Results
          </button>
        )}
      </div>

      {results.length > 0 && (
        <div className=" rounded-lg shadow-lg overflow-hidden">
          <h3 className="text-xl font-semibold p-4 ">Benchmark Results</h3>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="">
                <tr>
                  <th className="px-4 py-2 text-left">Provider</th>
                  <th className="px-4 py-2 text-left">Load Time (ms)</th>
                  <th className="px-4 py-2 text-left">Inference Time (ms)</th>
                  <th className="px-4 py-2 text-left">Memory (MB)</th>
                  <th className="px-4 py-2 text-left">CPU Cores</th>
                </tr>
              </thead>
              <tbody>
                {results.map((result, index) => (
                  <tr key={index} className="border-b">
                    <td className="px-4 py-2 font-medium">{result.provider}</td>
                    <td className="px-4 py-2">
                      {result.modelLoadTime.toFixed(1)}
                    </td>
                    <td className="px-4 py-2">
                      {result.inferenceTime.toFixed(1)}
                    </td>
                    <td className="px-4 py-2">
                      {result.memoryUsage.toFixed(1)}
                    </td>
                    <td className="px-4 py-2">{result.cpuCores}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="p-4  text-sm text-gray-600">
            <p>
              <strong>Note:</strong> Lower times = better performance. Results
              averaged over 5 runs.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
