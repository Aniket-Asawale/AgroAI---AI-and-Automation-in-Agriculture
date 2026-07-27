"""
Plant Disease Detection — Gradio UI

A simple web interface for testing plant disease detection.
Upload a leaf image and get disease prediction with treatment info.

Usage:
    cd Plant_Disease_Detection
    python gradio_app.py
"""

import io
import os

import gradio as gr
import requests

APP_TITLE = "AgroAI — Plant Disease Dashboard"

API_CANDIDATES = [
    os.getenv("DISEASE_API_URL", "").rstrip("/"),
    "https://api.agroaiapp.me/api/disease",
    "http://127.0.0.1:8080/api/disease",
    "http://127.0.0.1:8003",
]


def _api_endpoints():
    seen = set()
    for base in API_CANDIDATES:
        if base and base not in seen:
            seen.add(base)
            yield base


def _request_with_fallback(method, path, **kwargs):
    last_error = None
    for base_url in _api_endpoints():
        try:
            response = requests.request(method, f"{base_url}{path}", **kwargs)
            return response, base_url
        except requests.exceptions.RequestException as exc:
            last_error = exc
    raise last_error or requests.exceptions.ConnectionError("No disease API endpoints configured.")

def _fmt_list(items):
    if not items:
        return ""
    return "\n".join([f"- {x}" for x in items])

def _fmt_kv(d):
    if not d:
        return ""
    return "\n".join([f"- **{k}**: {v}" for k, v in d.items()])

def _badge(text, kind="info"):
    cls = f"badge badge-{kind}"
    return f"<span class='{cls}'>{text}</span>"

def _render_result(result: dict, api_base: str):
    pred = result.get("predicted_class", "unknown")
    conf = result.get("confidence", 0)
    crop = result.get("crop")
    dtype = result.get("disease_type")

    header = (
        f"<div class='result-header'>"
        f"<div class='result-title'>{pred}</div>"
        f"<div class='result-sub'>"
        f"{_badge(f'{conf}% confidence', 'good' if conf >= 60 else 'warn' if conf >= 35 else 'bad')}"
        f"{_badge(crop, 'info') if crop else ''}"
        f"{_badge(dtype, 'info') if dtype else ''}"
        f"</div>"
        f"<div class='result-meta'>Source: <code>{api_base}</code></div>"
        f"</div>"
    )

    symptoms = result.get("symptoms") or []
    precautions = result.get("precautions") or []
    chem = result.get("chemical_treatment") or {}
    org = result.get("organic_treatment") or {}

    details = "<div class='grid'>"
    details += "<div class='card'><div class='card-h'>Symptoms</div>"
    details += f"<div class='card-b'>{_fmt_list(symptoms) if symptoms else '—'}</div></div>"
    details += "<div class='card'><div class='card-h'>Precautions</div>"
    details += f"<div class='card-b'>{_fmt_list(precautions) if precautions else '—'}</div></div>"
    details += "<div class='card'><div class='card-h'>Chemical treatment</div>"
    details += f"<div class='card-b'>{_fmt_kv(chem) if chem else '—'}</div></div>"
    details += "<div class='card'><div class='card-h'>Organic treatment</div>"
    details += f"<div class='card-b'>{_fmt_kv(org) if org else '—'}</div></div>"
    details += "</div>"

    return header + details

def detect_disease(image, crop_model="all"):
    """
    Send image to the Plant Disease Detection API and return results.
    
    Args:
        image: PIL Image object
        crop_model: Crop model to use (all, corn, rice, wheat, millet, sugarcane)
    
    Returns:
        Tuple of (prediction_text, treatment_text, confidence_text)
    """
    if image is None:
        return "<div class='empty'>Upload a leaf image to start.</div>", {}, ""
    
    try:
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Send request to API
        files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}
        params = {'crop': crop_model}
        
        response, api_base = _request_with_fallback(
            "POST",
            "/detect",
            files=files,
            params=params,
            timeout=30,
        )
        
        if response.status_code == 200:
            result = response.json()
            html = _render_result(result, api_base)
            return html, result, ""
        else:
            error_msg = f"API Error from {api_base}: {response.status_code}"
            try:
                error_detail = response.json().get('detail', '')
                if error_detail:
                    error_msg += f"\n{error_detail}"
            except:
                pass
            return f"<div class='error'>{error_msg}</div>", {}, error_msg
            
    except requests.exceptions.ConnectionError:
        msg = "Cannot connect to the disease API. Start the gateway (8080) or the disease API (8003)."
        return f"<div class='error'>{msg}</div>", {}, msg
    except Exception as e:
        msg = f"Error: {str(e)}"
        return f"<div class='error'>{msg}</div>", {}, msg

def check_api_health():
    """Check if the API is running and healthy."""
    try:
        response, api_base = _request_with_fallback("GET", "/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            models = ", ".join(health.get("models_loaded", []) or [])
            return (
                f"{_badge('API healthy', 'good')} <span class='muted'>via</span> <code>{api_base}</code><br/>"
                f"<div class='health-grid'>"
                f"<div><div class='health-k'>Device</div><div class='health-v'>{health.get('device', 'unknown')}</div></div>"
                f"<div><div class='health-k'>Models</div><div class='health-v'>{models or '—'}</div></div>"
                f"<div><div class='health-k'>KB entries</div><div class='health-v'>{health.get('diseases_in_kb', 0)}</div></div>"
                f"</div>"
            )
        else:
            return f"<div class='error'>API returned {response.status_code} from <code>{api_base}</code></div>"
    except requests.exceptions.ConnectionError:
        return "<div class='error'>Disease API is not reachable. Start the gateway (8080) or disease API (8003).</div>"
    except Exception as e:
        return f"<div class='error'>Error checking API: {str(e)}</div>"

# Create Gradio interface
CSS = """
:root {
  --bg: #0b1220;
  --panel: rgba(255,255,255,0.06);
  --panel-2: rgba(255,255,255,0.09);
  --border: rgba(255,255,255,0.12);
  --text: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.65);
  --good: #22c55e;
  --warn: #f59e0b;
  --bad: #ef4444;
  --info: #60a5fa;
}

.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
body { background: radial-gradient(1200px 600px at 20% 0%, rgba(34,197,94,0.15), transparent 45%),
                 radial-gradient(900px 500px at 90% 10%, rgba(96,165,250,0.16), transparent 50%),
                 var(--bg) !important; }

.hero {
  border: 1px solid var(--border);
  background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
  border-radius: 16px;
  padding: 18px 18px 12px 18px;
  margin-bottom: 12px;
}
.hero h1 { margin: 0; color: var(--text); font-size: 22px; letter-spacing: 0.2px; }
.hero p { margin: 6px 0 0 0; color: var(--muted); }
.hero .row { display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }
.pill { border: 1px solid var(--border); background: var(--panel); border-radius: 999px; padding: 6px 10px; color: var(--muted); font-size: 12px; }
.pill strong { color: var(--text); font-weight: 600; }

.badge { display:inline-block; padding: 4px 10px; border-radius: 999px; font-size: 12px; border: 1px solid var(--border); background: var(--panel); color: var(--muted); margin-right: 6px; }
.badge-good { border-color: rgba(34,197,94,0.35); color: rgba(34,197,94,0.95); }
.badge-warn { border-color: rgba(245,158,11,0.35); color: rgba(245,158,11,0.95); }
.badge-bad  { border-color: rgba(239,68,68,0.35); color: rgba(239,68,68,0.95); }
.badge-info { border-color: rgba(96,165,250,0.35); color: rgba(96,165,250,0.95); }

.result-header { border: 1px solid var(--border); background: var(--panel); border-radius: 14px; padding: 14px; margin-bottom: 10px; }
.result-title { font-size: 18px; font-weight: 700; color: var(--text); }
.result-sub { margin-top: 8px; }
.result-meta { margin-top: 10px; color: var(--muted); font-size: 12px; }
.muted { color: var(--muted); }

.grid { display:grid; grid-template-columns: 1fr 1fr; gap: 10px; }
@media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
.card { border: 1px solid var(--border); background: var(--panel); border-radius: 14px; padding: 12px; }
.card-h { font-weight: 700; color: var(--text); margin-bottom: 8px; }
.card-b { color: var(--muted); }
.card-b strong { color: var(--text); }
.card-b code { color: var(--text); }

.empty { border: 1px dashed var(--border); background: rgba(255,255,255,0.03); border-radius: 14px; padding: 16px; color: var(--muted); }
.error { border: 1px solid rgba(239,68,68,0.35); background: rgba(239,68,68,0.06); border-radius: 14px; padding: 12px; color: rgba(255,255,255,0.88); white-space: pre-wrap; }

.health-grid { display:grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-top: 8px; }
@media (max-width: 900px) { .health-grid { grid-template-columns: 1fr; } }
.health-k { color: var(--muted); font-size: 12px; }
.health-v { color: var(--text); font-weight: 600; }
"""

theme = gr.themes.Soft(
    primary_hue="green",
    neutral_hue="slate",
)

with gr.Blocks(title=APP_TITLE, theme=theme, css=CSS) as demo:
    gr.HTML(
        f"""
        <div class="hero">
          <h1>{APP_TITLE}</h1>
          <p>Upload a leaf photo to detect disease and get treatment recommendations.</p>
          <div class="row">
            <div class="pill"><strong>Models</strong>: EfficientNet-B4</div>
            <div class="pill"><strong>Supported crops</strong>: corn, rice, wheat, millet, sugarcane</div>
            <div class="pill"><strong>Tip</strong>: use a close-up, well-lit leaf image</div>
          </div>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=5, min_width=420):
            image_input = gr.Image(
                type="pil",
                label="Leaf image",
                height=320,
                sources=["upload", "webcam"],
            )
            with gr.Row():
                crop_dropdown = gr.Dropdown(
                    choices=["all", "corn", "rice", "wheat", "millet", "sugarcane"],
                    value="all",
                    label="Crop model",
                    info="Pick a crop model if you know it; otherwise keep 'all'.",
                )
                detect_btn = gr.Button("Detect", variant="primary")
            with gr.Row():
                api_status_btn = gr.Button("API status", variant="secondary")
                clear_btn = gr.Button("Clear", variant="secondary")

            api_status_output = gr.HTML(value="", label="Status")

        with gr.Column(scale=7, min_width=520):
            result_html = gr.HTML(value="<div class='empty'>Upload a leaf image to start.</div>", label="Result")
            raw_json = gr.JSON(label="Raw response", value={})
            error_box = gr.Textbox(label="Error", value="", interactive=False, visible=False)
    
    with gr.Accordion("How to take a good photo", open=False):
        gr.HTML(
            """
            <div class="card-b">
              <ul>
                <li>Use daylight, avoid strong shadows.</li>
                <li>Fill most of the frame with a single leaf.</li>
                <li>Keep the leaf in focus; wipe water/dust if needed.</li>
                <li>Avoid busy backgrounds.</li>
              </ul>
            </div>
            """
        )
    
    # Event handlers
    detect_btn.click(
        fn=detect_disease,
        inputs=[image_input, crop_dropdown],
        outputs=[result_html, raw_json, error_box],
    )
    
    api_status_btn.click(
        fn=check_api_health,
        inputs=[],
        outputs=[api_status_output],
    )

    def _clear():
        return None, "<div class='empty'>Upload a leaf image to start.</div>", {}, ""

    clear_btn.click(
        fn=_clear,
        inputs=[],
        outputs=[image_input, result_html, raw_json, error_box],
    )
    
    # Auto-check API status on load
    demo.load(
        fn=check_api_health,
        inputs=[],
        outputs=[api_status_output],
    )

if __name__ == "__main__":
    print("=" * 60)
    print("  Plant Disease Detection — Gradio UI")
    print("=" * 60)
    print()
    print("  Starting Gradio interface...")
    print("  Local URL: http://127.0.0.1:7860")
    print()
    print("  Prerequisites:")
    print("    - Plant Disease Detection API running on port 8003")
    print("    - Run: start_disease.bat")
    print()
    print("=" * 60)
    
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
        show_error=True,
    )
