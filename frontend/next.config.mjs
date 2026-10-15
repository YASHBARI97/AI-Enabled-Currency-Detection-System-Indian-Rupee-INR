/** @type {import('next').NextConfig} */
// The browser calls /api/* on the Next.js origin; Next proxies to the FastAPI backend.
// This avoids CORS and keeps the backend URL out of client code.
const API_URL = process.env.API_URL || "http://127.0.0.1:8000";

const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_URL}/:path*` }];
  },
};

export default nextConfig;
