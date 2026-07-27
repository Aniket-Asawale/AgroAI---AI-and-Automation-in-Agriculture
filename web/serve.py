"""
Simple HTTP server for the AgroModules landing page.
Serves files from this directory on port 8505.
"""
import http.server
import socketserver
import os
import sys
from urllib.parse import urlsplit

PORT = 8505
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

os.chdir(DIRECTORY)

# Content Security Policy — allowlists only the CDNs and the Supabase
# backend the site actually uses. 'unsafe-inline' for scripts/styles is
# required because the importmap and small page scripts are inline; the
# rest is locked down to mitigate XSS, clickjacking and data exfiltration.
CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' https://unpkg.com "
    "https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com data:; "
    "img-src 'self' data:; "
    "connect-src 'self' https://*.supabase.co wss://*.supabase.co https://cdn.jsdelivr.net; "
    "frame-src https://accounts.google.com; "
    "base-uri 'self'; form-action 'self'; object-src 'none'; frame-ancestors 'none'"
)

SECURITY_HEADERS = {
    "Content-Security-Policy": CSP,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Cross-Origin-Opener-Policy": "same-origin",
}

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlsplit(self.path)
        # Never expose server-side source (e.g. this file) to the public.
        if parsed.path.lower().endswith((".py", ".pyc")):
            self.send_error(404, "File not found")
            return
        redirects = {
            "/plant-disease/dashboard": "https://disease-dashboard.agroaiapp.me/",
            "/plant-disease/dashboard/": "https://disease-dashboard.agroaiapp.me/",
            "/sensor/dashboard": "https://sensor-dashboard.agroaiapp.me/",
            "/sensor/dashboard/": "https://sensor-dashboard.agroaiapp.me/",
        }
        target = redirects.get(parsed.path)
        if target:
            self.send_response(302)
            self.send_header("Location", target)
            self.end_headers()
            return
        super().do_GET()

    def log_message(self, format, *args):
        # Only log client/server errors (4xx/5xx), not every request.
        status = str(args[1]) if len(args) > 1 else ""
        if status.startswith("4") or status.startswith("5"):
            super().log_message(format, *args)

    def end_headers(self):
        # Security headers on every response.
        for name, value in SECURITY_HEADERS.items():
            self.send_header(name, value)
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

if __name__ == "__main__":
    with socketserver.TCPServer(("127.0.0.1", PORT), QuietHandler) as httpd:
        print(f"Homepage serving at http://127.0.0.1:{PORT}")
        print(f"Directory: {DIRECTORY}")
        sys.stdout.flush()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
            httpd.shutdown()
