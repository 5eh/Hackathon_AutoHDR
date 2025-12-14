import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Remove the unrecognized 'dev' key.
  // We will force webpack via the 'package.json' script instead.

  webpack: (config, { isServer }) => {
    // Ignore node-specific modules when bundling for the browser (client-side)
    // and provide fallbacks.
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
        crypto: false,
        stream: false,
      };
    }
    return config;
  },
};

export default nextConfig;
