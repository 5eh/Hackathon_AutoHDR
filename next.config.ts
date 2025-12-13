import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Minimal config - let Turbopack handle most things automatically
  turbopack: {},

  // Allow loading .onnx files from public directory
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          {
            key: "Cross-Origin-Embedder-Policy",
            value: "require-corp",
          },
          {
            key: "Cross-Origin-Opener-Policy",
            value: "same-origin",
          },
        ],
      },
    ];
  },
};

export default nextConfig;
