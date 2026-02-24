/**
 * API base URL for backend requests.
 * - Development: empty string → relative /api/* (Vite proxy forwards to localhost:8000)
 * - Production: VITE_API_URL from .env.production → full URL
 */
export const API_BASE = (import.meta.env.VITE_API_URL as string) ?? "";
