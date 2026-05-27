import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Required for the multi-stage Docker build (copies only the minimal
  // runtime files into the final image via .next/standalone/).
  output: "standalone",
};

export default nextConfig;
