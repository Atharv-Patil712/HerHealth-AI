/// <reference types="vite/client" />

export const APP_NAME = "HerHealth AI";
export const BRAND_LINE = "Your health, understood.";
export const SUPPORTING_MESSAGE =
  "A gentler way to understand your health, notice patterns and take informed next steps.";

// In development, relative API requests are handled by the Vite proxy.
// Production requests go to the deployed API unless explicitly overridden.
export const API_BASE = (
  import.meta.env.VITE_API_BASE ??
  (import.meta.env.DEV ? "" : "https://herhealth-ai-api.onrender.com")
).replace(/\/$/, "");