/** @type {import('next').NextConfig} */
const nextConfig = {
  // Static export to frontend/out, served by the FastAPI backend at `/` (backend/app/main.py).
  output: 'export',
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
}

export default nextConfig
