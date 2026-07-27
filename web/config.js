// ─────────────────────────────────────────────────────────────
// AgroAI — front-end configuration.
//
// Supabase credentials (safe to expose — anon/publishable key only).
// All data access is enforced server-side by Row Level Security (RLS).
// Never put the "service_role" / secret key here.
// ─────────────────────────────────────────────────────────────

export const SUPABASE_URL      = "https://wmvttcrsadhuskfhmybn.supabase.co";
export const SUPABASE_ANON_KEY = "sb_publishable_LjJD-aCNGYomqYEqG-Duig_fYBDpAYw";

// Production site URL for auth redirects (use window.location.origin for local dev)
export const SITE_URL = "https://agroaiapp.me";

// Auth is enabled since both values are present.
export const AUTH_ENABLED = Boolean(SUPABASE_URL && SUPABASE_ANON_KEY);
