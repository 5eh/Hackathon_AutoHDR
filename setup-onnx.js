// setup-onnx.js
const fs = require("fs");
const path = require("path");

const wasmSource = path.join(__dirname, "node_modules/onnxruntime-web/dist");
const wasmDest = path.join(__dirname, "public");

// Ensure public directory exists
if (!fs.existsSync(wasmDest)) {
  fs.mkdirSync(wasmDest, { recursive: true });
}

console.log("Checking ONNX Runtime dist folder...");

// List of ONNX Runtime files needed for web/wasm execution
const filesToCopy = [
  // Core WASM and worker files
  "ort-wasm-simd-threaded.wasm",
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.jsep.mjs",

  // Main JavaScript bundle files
  "ort.webgpu.min.js",
  "ort.min.js",
  // Removed "ort.es6.min.js" to fix the warning.
];

let copiedCount = 0;
filesToCopy.forEach((file) => {
  const srcPath = path.join(wasmSource, file);
  const destPath = path.join(wasmDest, file);

  if (fs.existsSync(srcPath)) {
    fs.copyFileSync(srcPath, destPath);
    console.log(`✓ Copied ${file}`);
    copiedCount++;
  } else {
    console.warn(
      `⚠ Warning: ${file} not found in node_modules/onnxruntime-web/dist`,
    );
  }
});

console.log(
  `\nONNX Runtime Web setup complete! Copied ${copiedCount} files to public/`,
);
