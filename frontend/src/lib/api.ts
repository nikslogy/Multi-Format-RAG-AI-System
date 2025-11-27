// API configuration
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// API endpoints
export const API_ENDPOINTS = {
  upload: `${API_BASE_URL}/api/upload`,
  query: `${API_BASE_URL}/api/query`,
  documents: `${API_BASE_URL}/api/documents`,
  stats: `${API_BASE_URL}/api/stats`,
  health: `${API_BASE_URL}/health`,
} as const;
