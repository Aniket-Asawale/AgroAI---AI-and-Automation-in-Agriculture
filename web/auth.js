// ─────────────────────────────────────────────────────────────
// AgroAI — auth.js
// Supabase email/password + Google OAuth.
// Uses the publishable anon key — safe in client code.
// ─────────────────────────────────────────────────────────────
import { createClient } from "https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2/+esm";
import { SUPABASE_URL, SUPABASE_ANON_KEY, AUTH_ENABLED, SITE_URL } from "./config.js";

let _client = null;

export function getSupabase() {
  if (!AUTH_ENABLED) return null;
  if (!_client) {
    _client = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
      auth: {
        persistSession: true,
        autoRefreshToken: true,
        detectSessionInUrl: true,
      },
    });
  }
  return _client;
}

export function authReady() {
  return AUTH_ENABLED;
}

export async function getUser() {
  const sb = getSupabase();
  if (!sb) return null;
  const { data: { user } } = await sb.auth.getUser();
  return user ?? null;
}

export async function getSession() {
  const sb = getSupabase();
  if (!sb) return null;
  const { data: { session } } = await sb.auth.getSession();
  return session ?? null;
}

export async function signUp(email, password, metadata = {}) {
  const sb = getSupabase();
  if (!sb) throw new Error("Auth is not configured.");
  return sb.auth.signUp({
    email,
    password,
    options: {
      data: metadata,
      emailRedirectTo: `${SITE_URL}/dashboard.html`,
    },
  });
}

export async function signInWithPassword(email, password) {
  const sb = getSupabase();
  if (!sb) throw new Error("Auth is not configured.");
  return sb.auth.signInWithPassword({ email, password });
}

export async function signInWithGoogle() {
  const sb = getSupabase();
  if (!sb) throw new Error("Auth is not configured.");
  return sb.auth.signInWithOAuth({
    provider: "google",
    options: { redirectTo: `${SITE_URL}/dashboard.html` },
  });
}

export async function signOut() {
  const sb = getSupabase();
  if (sb) await sb.auth.signOut();
  window.location.href = `${SITE_URL}/index.html`;
}

// Redirect to login if not authenticated — used by gated pages.
export async function requireAuth() {
  if (!AUTH_ENABLED) {
    window.location.href = `${SITE_URL}/login.html`;
    return null;
  }
  const user = await getUser();
  if (!user) {
    window.location.href = `${SITE_URL}/login.html`;
    return null;
  }
  return user;
}

// Update navbar account button based on auth state.
async function updateNavAccount() {
  const el = document.getElementById("nav-account");
  if (!el) return;
  if (!AUTH_ENABLED) return; // keep "Sign In"
  const user = await getUser();
  if (user) {
    el.textContent = "Dashboard";
    el.setAttribute("href", `${SITE_URL}/dashboard.html`);
  } else {
    el.textContent = "Sign In";
    el.setAttribute("href", `${SITE_URL}/login.html`);
  }
}

if (typeof document !== "undefined") {
  document.addEventListener("DOMContentLoaded", updateNavAccount);
  const sb = getSupabase();
  if (sb) sb.auth.onAuthStateChange(() => updateNavAccount());
}
