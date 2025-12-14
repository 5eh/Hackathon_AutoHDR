const fs = require("fs");
const path = require("path");

// Find ONNX Runtime WASM files in node_modules
const findWasmFiles = () => {
  const possiblePaths = [
    path.join(__dirname, "node_modules/onnxruntime-web/dist"),
    path.join(__dirname, "node_modules/onnxruntime-web/lib"),
    path.join(__dirname, "node_modules/onnxruntime-web"),
  ];

  for (const basePath of possiblePaths) {
    if (fs.existsSync(basePath)) {
      console.log(`Checking directory: ${basePath}`);
      const files = fs.readdirSync(basePath);
      console.log(`Files found: ${files.join(", ")}`);

      // Look for WASM files
      const wasmFiles = files.filter((file) => file.endsWith(".wasm"));
      if (wasmFiles.length > 0) {
        console.log(`WASM files found: ${wasmFiles.join(", ")}`);
        return { path: basePath, files: wasmFiles };
      }
    }
  }

  return null;
};

// Copy ONNX Runtime WASM files to public directory
const targetDir = path.join(__dirname, "public");

// Ensure public directory exists
if (!fs.existsSync(targetDir)) {
  fs.mkdirSync(targetDir, { recursive: true });
}

const wasmInfo = findWasmFiles();

if (wasmInfo) {
  console.log(`Copying WASM files from: ${wasmInfo.path}`);

  wasmInfo.files.forEach((file) => {
    const sourcePath = path.join(wasmInfo.path, file);
    const targetPath = path.join(targetDir, file);

    try {
      fs.copyFileSync(sourcePath, targetPath);
      console.log(`✓ Copied ${file} to public directory`);
    } catch (error) {
      console.error(`✗ Failed to copy ${file}:`, error.message);
    }
  });
} else {
  console.log(
    "No WASM files found. ONNX Runtime might work without them or they may be bundled differently.",
  );
  console.log("This is normal for some versions of onnxruntime-web.");
}

console.log("ONNX Runtime setup complete!");
