/** @type {import('next').NextConfig} */
const nextConfig = {
  // Optimize for Vercel serverless
  serverExternalPackages: [],
  
  // Enable static optimization where possible
  output: undefined, // Let Vercel handle this
  
  // Webpack config for handling any edge cases
  webpack: (config, { isServer }) => {
    if (!isServer) {
      // Don't bundle server-only modules on client
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
      };
    }
    return config;
  },
};

export default nextConfig;
