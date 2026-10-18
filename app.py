import base64
import html
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path

import requests
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image as PILImage
from torchvision import models, transforms

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as PDFImage,
)

# =========================================================
# CONFIGURATION
# =========================================================

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "outputs" / "best_model.pth"

OLLAMA_BASE_URL = "http://host.docker.internal:11434"
OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_MODEL = "qwen2.5vl:3b"

IMAGE_SIZE = 160
DEVICE = torch.device("cpu")

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]

CLASS_LABELS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}

# Colors used only to tell the classes apart visually in the UI.
CLASS_COLORS = {
    "glioma": "#e07a5f",
    "meningioma": "#8d6cc9",
    "notumor": "#2a9d8f",
    "pituitary": "#3d7ccf",
}

# ImageNet normalization used with pretrained MobileNetV3
TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="NeuroScan AI | MRI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CUSTOM STYLING
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #0f2438;
        --navy: #0b2239;
        --navy-2: #13395a;
        --teal: #14a3a8;
        --teal-dark: #0e7c86;
        --teal-soft: #e3f6f6;
        --page: #f3f7fb;
        --card: #ffffff;
        --muted: #64788b;
        --line: #e2eaf2;
        --shadow: 0 8px 28px rgba(11, 34, 57, .06);
    }

    html, body, [class*="css"], .stApp, .stMarkdown, button, input {
        font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(900px 420px at 100% -10%, rgba(20,163,168,.10), transparent 60%),
            radial-gradient(700px 360px at -10% 0%, rgba(19,57,90,.07), transparent 60%),
            var(--page);
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }
    .block-container {
        max-width: 1280px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b2239 0%, #0f2e4a 100%);
        border-right: 1px solid rgba(255,255,255,.06);
    }
    [data-testid="stSidebar"] * { color: #e8f1f8; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.12); }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #9fb6c8; }

    .side-brand { display:flex; align-items:center; gap:12px; margin-bottom:4px; }
    .side-logo {
        width: 42px; height: 42px; border-radius: 12px;
        display:flex; align-items:center; justify-content:center; font-size:22px;
        background: linear-gradient(135deg, #14a3a8, #3d7ccf);
        box-shadow: 0 6px 18px rgba(20,163,168,.35);
    }
    .side-name { font-size: 18px; font-weight: 800; letter-spacing: -.02em; }
    .side-sub { font-size: 12px; color: #9fb6c8 !important; }
    .side-title {
        font-size: 11px; font-weight: 700; letter-spacing: .12em;
        text-transform: uppercase; color: #7fa0b8 !important; margin: 4px 0 10px 0;
    }
    .chip {
        display:flex; align-items:center; gap:10px; padding: 10px 12px;
        border-radius: 12px; margin-bottom: 8px; font-size: 13px; font-weight: 600;
        background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.08);
    }
    .chip .dot { width: 9px; height: 9px; border-radius: 50%; flex: none; }
    .chip.ok .dot   { background: #34d399; box-shadow: 0 0 0 4px rgba(52,211,153,.18); }
    .chip.warn .dot { background: #fbbf24; box-shadow: 0 0 0 4px rgba(251,191,36,.18); }
    .chip.bad .dot  { background: #f87171; box-shadow: 0 0 0 4px rgba(248,113,113,.18); }
    .info-row {
        display:flex; justify-content:space-between; gap:10px;
        padding: 8px 0; font-size: 13px;
        border-bottom: 1px dashed rgba(255,255,255,.10);
    }
    .info-row span:first-child { color: #9fb6c8 !important; }
    .info-row span:last-child { font-weight: 700; text-align: right; }

    /* ---------- Hero ---------- */
    .hero {
        position: relative; overflow: hidden;
        background: linear-gradient(120deg, #0b2239 0%, #13395a 55%, #0e7c86 120%);
        border-radius: 22px; padding: 34px 38px; margin-bottom: 16px;
        box-shadow: 0 18px 40px rgba(11,34,57,.22);
    }
    .hero::after {
        content: ""; position: absolute; right: -80px; top: -80px;
        width: 320px; height: 320px; border-radius: 50%;
        background: radial-gradient(circle, rgba(20,163,168,.45), transparent 65%);
    }
    .hero-top { display:flex; align-items:center; gap:12px; margin-bottom: 18px; position: relative; z-index: 1; }
    .hero-mark {
        width: 44px; height: 44px; border-radius: 13px; font-size: 23px;
        display:flex; align-items:center; justify-content:center;
        background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.18);
    }
    .hero-brand { color:#fff; font-size: 17px; font-weight: 800; letter-spacing: -.01em; }
    .hero-tag {
        margin-left: auto; color: #bfeff1; font-size: 11px; font-weight: 700;
        letter-spacing: .08em; text-transform: uppercase;
        border: 1px solid rgba(255,255,255,.22); border-radius: 999px; padding: 7px 13px;
        background: rgba(255,255,255,.07);
    }
    .hero h1 {
        position: relative; z-index: 1; color: #fff !important;
        font-size: 36px; line-height: 1.15; letter-spacing: -.035em;
        margin: 0 0 10px 0; padding: 0;
    }
    .hero p {
        position: relative; z-index: 1; color: #c6d8e6; font-size: 15px;
        line-height: 1.7; margin: 0 0 20px 0; max-width: 720px;
    }
    .badges { position: relative; z-index: 1; display:flex; flex-wrap:wrap; gap: 8px; }
    .badge {
        color: #e8f6f7; font-size: 12px; font-weight: 600; padding: 7px 12px;
        border-radius: 999px; background: rgba(255,255,255,.10);
        border: 1px solid rgba(255,255,255,.16);
    }

    /* ---------- Notice ---------- */
    .notice {
        display:flex; gap: 12px; align-items:flex-start;
        background: #fff8e8; border: 1px solid #f6e2b0; border-left: 4px solid #f0b429;
        border-radius: 14px; padding: 14px 16px; margin-bottom: 18px;
        color: #6b4e08; font-size: 13.5px; line-height: 1.6;
    }
    .notice b { color: #5a4106; }

    /* ---------- Stepper ---------- */
    .stepper { display:grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px; }
    .step {
        display:flex; align-items:center; gap: 12px; padding: 13px 15px;
        background: var(--card); border: 1px solid var(--line); border-radius: 14px;
        box-shadow: 0 3px 12px rgba(11,34,57,.03);
    }
    .step-num {
        width: 32px; height: 32px; border-radius: 10px; flex: none;
        display:flex; align-items:center; justify-content:center;
        font-size: 13px; font-weight: 800; background: #eef3f8; color: #7a8da0;
    }
    .step-label { font-size: 13px; font-weight: 700; color: #5d7184; line-height: 1.25; }
    .step-state { font-size: 11px; color: #94a5b5; font-weight: 500; }
    .step.active { border-color: #9edcdf; background: #f4fcfc; }
    .step.active .step-num { background: var(--teal); color: #fff; }
    .step.active .step-label { color: var(--navy); }
    .step.done .step-num { background: #d8f3ec; color: #1b8a70; }
    .step.done .step-label { color: var(--navy); }
    .step.done .step-state { color: #1b8a70; }

    /* ---------- Section heading ---------- */
    .section-heading { display:flex; align-items:center; gap: 12px; margin: 4px 0 14px 0; }
    .section-index {
        width: 34px; height: 34px; border-radius: 11px; font-size: 12px; font-weight: 800;
        display:flex; align-items:center; justify-content:center;
        background: var(--teal-soft); color: var(--teal-dark);
    }
    .section-title { font-size: 19px; font-weight: 800; color: var(--navy); letter-spacing: -.02em; }
    .section-sub { font-size: 12.5px; color: var(--muted); margin-top: 1px; }

    /* ---------- Cards ---------- */
    .result-banner {
        position: relative; overflow: hidden;
        background: var(--card); border: 1px solid var(--line);
        border-radius: 18px; padding: 22px 24px; margin-bottom: 16px;
        box-shadow: var(--shadow);
    }
    .result-banner::before {
        content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 6px;
        background: var(--accent, var(--teal));
    }
    .result-eyebrow {
        color: var(--muted); font-size: 11px; font-weight: 800;
        letter-spacing: .12em; text-transform: uppercase; margin-bottom: 8px;
    }
    .result-row { display:flex; align-items:flex-end; justify-content:space-between; gap: 14px; flex-wrap: wrap; }
    .result-label { color: var(--navy); font-size: 30px; font-weight: 800; letter-spacing: -.03em; line-height: 1.1; }
    .result-score { font-size: 34px; font-weight: 800; color: var(--accent, var(--teal)); letter-spacing: -.03em; line-height: 1; }
    .result-score small { display:block; font-size: 11px; font-weight: 600; color: var(--muted); letter-spacing: 0; margin-top: 4px; text-align: right; }
    .small-note { color: var(--muted); font-size: 12.5px; line-height: 1.6; margin: 12px 0 0 0; }

    .bars-card {
        background: var(--card); border: 1px solid var(--line); border-radius: 18px;
        padding: 20px 22px; box-shadow: var(--shadow);
    }
    .bars-title { font-size: 13px; font-weight: 800; color: var(--navy); margin-bottom: 14px; letter-spacing: .02em; }
    .bar-row { margin-bottom: 14px; }
    .bar-row:last-child { margin-bottom: 0; }
    .bar-head { display:flex; justify-content:space-between; font-size: 13.5px; font-weight: 600; color: var(--ink); margin-bottom: 6px; }
    .bar-head .pct { font-variant-numeric: tabular-nums; color: var(--muted); font-weight: 700; }
    .bar-head .dotc { display:inline-block; width: 9px; height: 9px; border-radius: 3px; margin-right: 8px; }
    .bar-track { height: 10px; background: #eaf0f6; border-radius: 99px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 99px; }

    .empty-state {
        text-align: center; padding: 46px 24px; background: var(--card);
        border: 1.5px dashed #c5d5e2; border-radius: 18px; color: var(--muted);
    }
    .empty-state .big { font-size: 38px; margin-bottom: 8px; }
    .empty-state .t { font-weight: 800; color: var(--navy); font-size: 16px; margin-bottom: 4px; }
    .empty-state .d { font-size: 13px; line-height: 1.6; }

    /* ---------- Streamlit widgets ---------- */
    [data-testid="stFileUploader"] section {
        background: #fff; border: 1.5px dashed #a9c3d4; border-radius: 16px; padding: 18px;
    }
    [data-testid="stFileUploader"] section:hover { border-color: var(--teal); background: #f7fcfc; }
    [data-testid="stImage"] img {
        border-radius: 14px; border: 1px solid var(--line); box-shadow: var(--shadow);
    }
    [data-testid="stAlert"] { border-radius: 12px; }

    div.stButton > button, div.stDownloadButton > button {
        min-height: 46px; border-radius: 12px; font-weight: 700; font-size: 14px;
        border: 1px solid #d3e0ea; background: #fff; color: var(--navy);
        transition: all 160ms ease;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        border-color: var(--teal); color: var(--teal-dark); transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(20,163,168,.15);
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #14a3a8, #0e7c86);
        color: #fff; border: none; box-shadow: 0 8px 20px rgba(14,124,134,.28);
    }
    div.stButton > button[kind="primary"]:hover { color: #fff; filter: brightness(1.06); }
    div.stButton > button:disabled { opacity: .5; transform: none; box-shadow: none; }

    /* Tabs */
    [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); }
    [data-baseweb="tab"] { font-weight: 700; padding: 10px 16px; border-radius: 10px 10px 0 0; }
    [data-baseweb="tab-highlight"] { background: var(--teal) !important; height: 3px; }

    .report-box {
        background: var(--card); border: 1px solid var(--line); border-radius: 16px;
        padding: 6px 22px 10px 22px; box-shadow: var(--shadow); margin: 12px 0;
    }

    .app-footer {
        text-align: center; color: var(--muted); font-size: 12px; line-height: 1.7;
        padding: 18px 10px 4px 10px; border-top: 1px solid var(--line); margin-top: 26px;
    }

    @media (max-width: 900px) {
        .stepper { grid-template-columns: repeat(2, 1fr); }
        .hero { padding: 24px 22px; }
        .hero h1 { font-size: 28px; }
        .hero-tag { display: none; }
    }
    @media (max-width: 560px) {
        .stepper { grid-template-columns: 1fr; }
        .result-label { font-size: 24px; }
        .result-score { font-size: 28px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def html_block(markup):
    """Remove indentation so Streamlit's Markdown does not treat HTML as a code block."""
    return "\n".join(line.strip() for line in markup.strip().splitlines() if line.strip())


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_classifier():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    model = models.mobilenet_v3_small(weights=None)

    # The training script uses a four-class classification head.
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        len(CLASS_NAMES),
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    # Support either a raw state_dict or a wrapped checkpoint.
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError("The saved model checkpoint is not valid.")

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


def classify_mri(image):
    model = load_classifier()

    image = image.convert("RGB")
    input_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    results = [
        {
            "class_name": CLASS_NAMES[i],
            "label": CLASS_LABELS[CLASS_NAMES[i]],
            "score": float(probabilities[i].item()),
        }
        for i in range(len(CLASS_NAMES))
    ]

    results.sort(key=lambda item: item["score"], reverse=True)

    return results


# =========================================================
# QWEN REPORT GENERATION THROUGH OLLAMA
# =========================================================

def generate_report(image_bytes, results):
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    prediction_lines = "\n".join(
        f"- {item['label']}: {item['score'] * 100:.2f}%"
        for item in results
    )

    predicted_label = results[0]["label"]

    system_prompt = """
You are an AI assistant helping explain the output of an
experimental educational brain MRI image-classification project.

Rules:
- Do not claim to diagnose a patient.
- Do not describe the predicted class as a confirmed disease.
- Explain that model scores are not calibrated disease probabilities.
- The image may be a cropped or 2D dataset image, not a complete MRI study.
- Do not invent image findings, patient symptoms, medical history,
  or examination results.
- Do not recommend CT, PET, repeat imaging, medication, or treatment.
- If clinical interpretation is needed, say that a qualified clinician
  should review the complete MRI study and relevant clinical information.
- Distinguish the classifier output from your own visual observations.
- If the image is unclear or cannot be reliably interpreted, say so.
- Keep the report concise and use these headings:
  1. AI Classifier Output
  2. Image Review Limitations
  3. Information Needed for Clinical Interpretation
  4. Safety Disclaimer
"""

    user_message = f"""
The experimental classifier's highest-scoring class is:
{predicted_label}

Classifier scores:
{prediction_lines}

Review the attached image cautiously and produce an educational report.
Do not change or reinterpret the numerical classifier scores.
"""

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_message,
                "images": [image_base64],
            },
        ],
        "options": {
            "temperature": 0.2,
            "num_predict": 450,
        },
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=300,
    )
    response.raise_for_status()

    response_data = response.json()
    report = response_data.get("message", {}).get("content", "")

    if not report.strip():
        raise ValueError("Ollama returned an empty report.")

    return report.strip()


# =========================================================
# PDF REPORT GENERATION
# =========================================================

def create_pdf_report(image_bytes, results, report_text=""):
    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=42,
        leftMargin=42,
        topMargin=42,
        bottomMargin=42,
        title="NeuroScan AI MRI Analysis Report",
        author="NeuroScan AI",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "NeuroScanTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=23,
        leading=28,
        textColor=colors.HexColor("#12304A"),
        alignment=TA_CENTER,
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        "NeuroScanSubtitle",
        parent=styles["Normal"],
        fontSize=11,
        leading=16,
        textColor=colors.HexColor("#526477"),
        alignment=TA_CENTER,
        spaceAfter=18,
    )

    body_style = ParagraphStyle(
        "NeuroScanBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
        spaceAfter=7,
    )

    story = []

    story.append(Paragraph("NeuroScan AI", title_style))
    story.append(
        Paragraph("Brain MRI Classification Report", subtitle_style)
    )

    story.append(Paragraph(
        f"<b>Generated:</b> "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        body_style,
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("MRI Image", styles["Heading2"]))

    try:
        original_image = PILImage.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        image_buffer = BytesIO()
        original_image.save(image_buffer, format="JPEG")
        image_buffer.seek(0)

        pdf_image = PDFImage(
            image_buffer,
            width=3.5 * inch,
            height=3.5 * inch,
            kind="proportional",
        )

        story.append(pdf_image)
        story.append(Spacer(1, 14))

    except Exception:
        story.append(Paragraph(
            "The uploaded image could not be embedded in the PDF.",
            body_style,
        ))

    story.append(Paragraph(
        "Classifier Results",
        styles["Heading2"],
    ))

    table_data = [["Class", "Model Score"]]

    for item in results:
        table_data.append([
            item["label"],
            f"{item['score'] * 100:.2f}%",
        ])

    results_table = Table(
        table_data,
        colWidths=[3.6 * inch, 1.4 * inch],
        repeatRows=1,
    )

    results_table.setStyle(TableStyle([
        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.HexColor("#12304A"),
        ),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D6E0E8")),
        ("PADDING", (0, 0), (-1, -1), 8),
        (
            "ROWBACKGROUNDS",
            (0, 1),
            (-1, -1),
            [colors.white, colors.HexColor("#F0F5F8")],
        ),
    ]))

    story.append(results_table)
    story.append(Spacer(1, 18))

    if report_text.strip():
        story.append(Paragraph(
            "AI-Generated Educational Report",
            styles["Heading2"],
        ))

        # Escape HTML special characters before using Paragraph.
        for line in report_text.splitlines():
            cleaned_line = html.escape(line.strip())

            if cleaned_line:
                story.append(
                    Paragraph(cleaned_line, body_style)
                )

    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "Important Safety Disclaimer",
        styles["Heading2"],
    ))

    disclaimer = (
        "This report is generated by an experimental AI system for "
        "educational and research purposes only. Model scores are not "
        "confirmed disease probabilities. The model has not been "
        "established as a clinically validated diagnostic system. "
        "Do not use this report to diagnose a patient or make treatment "
        "decisions. A qualified medical professional must interpret "
        "the complete MRI study in its clinical context."
    )

    story.append(Paragraph(disclaimer, body_style))

    document.build(story)
    buffer.seek(0)

    return buffer.getvalue()


# =========================================================
# UI HELPERS
# =========================================================

@st.cache_data(ttl=20, show_spinner=False)
def check_ollama():
    """Return (state, text) where state is ok / warn / bad."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)

        if not response.ok:
            return "warn", "Ollama status unknown"

        available_models = [
            item.get("name", "")
            for item in response.json().get("models", [])
        ]

        if any(
            name.split(":")[0] == OLLAMA_MODEL.split(":")[0]
            for name in available_models
        ):
            return "ok", "Ollama connected"

        return "warn", f"Model missing: {OLLAMA_MODEL}"

    except requests.RequestException:
        return "bad", "Ollama not responding"


def chip(state, text):
    return f'<div class="chip {state}"><span class="dot"></span>{html.escape(text)}</div>'


def section_heading(index, title, subtitle=""):
    sub = f'<div class="section-sub">{html.escape(subtitle)}</div>' if subtitle else ""
    return html_block(
        f"""
        <div class="section-heading">
            <span class="section-index">{index}</span>
            <div><div class="section-title">{html.escape(title)}</div>{sub}</div>
        </div>
        """
    )


def render_stepper(has_image, has_results, has_report):
    steps = [
        ("Upload", has_image, "Add an MRI image"),
        ("Classify", has_results, "Run MobileNetV3"),
        ("Report", has_report, "Qwen2.5-VL summary"),
        ("Export", has_results, "TXT or PDF"),
    ]

    parts = []
    active_found = False

    for number, (label, done, hint) in enumerate(steps, start=1):
        if done:
            css, state_text, icon = "done", "Completed", "✓"
        elif not active_found:
            css, state_text, icon = "active", "Current step", str(number)
            active_found = True
        else:
            css, state_text, icon = "", hint, str(number)

        # Export is only "done" visually once a report exists too.
        if label == "Export":
            css, state_text, icon = (
                ("done", "Ready to download", "✓") if has_report
                else ("active", "PDF available", str(number)) if has_results
                else ("", hint, str(number))
            )

        parts.append(
            f'<div class="step {css}"><div class="step-num">{icon}</div>'
            f'<div><div class="step-label">{label}</div>'
            f'<div class="step-state">{html.escape(state_text)}</div></div></div>'
        )

    st.markdown(
        f'<div class="stepper">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def render_results(results):
    top = results[0]
    accent = CLASS_COLORS.get(top["class_name"], "#14a3a8")

    st.markdown(
        html_block(
            f"""
            <div class="result-banner" style="--accent:{accent}">
                <div class="result-eyebrow">Top model output</div>
                <div class="result-row">
                    <div class="result-label">{html.escape(top["label"])}</div>
                    <div class="result-score">{top["score"] * 100:.1f}%<small>model score</small></div>
                </div>
                <p class="small-note">Classifier score only — not a calibrated disease probability or a diagnosis.</p>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    rows = []
    for item in results:
        color = CLASS_COLORS.get(item["class_name"], "#14a3a8")
        pct = max(0.0, min(1.0, item["score"])) * 100
        rows.append(
            f'<div class="bar-row"><div class="bar-head">'
            f'<span><span class="dotc" style="background:{color}"></span>{html.escape(item["label"])}</span>'
            f'<span class="pct">{pct:.2f}%</span></div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.2f}%;background:{color}"></div></div></div>'
        )

    st.markdown(
        f'<div class="bars-card"><div class="bars-title">SCORES BY CLASS</div>{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )


def empty_state(icon, title, description):
    st.markdown(
        html_block(
            f"""
            <div class="empty-state">
                <div class="big">{icon}</div>
                <div class="t">{html.escape(title)}</div>
                <div class="d">{html.escape(description)}</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


# =========================================================
# SESSION STATE
# =========================================================

if "mri_results" not in st.session_state:
    st.session_state["mri_results"] = None

if "image_bytes" not in st.session_state:
    st.session_state["image_bytes"] = None

if "mri_report" not in st.session_state:
    st.session_state["mri_report"] = ""

if "analysis_error" not in st.session_state:
    st.session_state["analysis_error"] = ""


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown(
        html_block(
            """
            <div class="side-brand">
                <div class="side-logo">🧠</div>
                <div>
                    <div class="side-name">NeuroScan AI</div>
                    <div class="side-sub">Brain MRI Analysis Assistant</div>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown('<div class="side-title">System status</div>', unsafe_allow_html=True)

    if MODEL_PATH.exists():
        st.markdown(chip("ok", "Classifier checkpoint found"), unsafe_allow_html=True)
    else:
        st.markdown(chip("bad", "Classifier checkpoint missing"), unsafe_allow_html=True)

    ollama_state, ollama_text = check_ollama()
    st.markdown(chip(ollama_state, ollama_text), unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="side-title">Project information</div>', unsafe_allow_html=True)

    st.markdown(
        html_block(
            """
            <div class="info-row"><span>Classifier</span><span>MobileNetV3-Small</span></div>
            <div class="info-row"><span>Report model</span><span>Qwen2.5-VL 3B</span></div>
            <div class="info-row"><span>Device</span><span>CPU</span></div>
            <div class="info-row"><span>Classes</span><span>4</span></div>
            """
        ),
        unsafe_allow_html=True,
    )

    st.divider()

    st.caption(
        "Research and educational prototype. "
        "Not intended for clinical diagnosis."
    )


# =========================================================
# HERO + NOTICE
# =========================================================

st.markdown(
    html_block(
        """
        <div class="hero">
            <div class="hero-top">
                <div class="hero-mark">🧠</div>
                <div class="hero-brand">NeuroScan AI</div>
                <div class="hero-tag">● Research prototype</div>
            </div>
            <h1>Brain MRI analysis,<br>explained clearly.</h1>
            <p>
                Upload a 2D MRI image to review the experimental MobileNetV3 classification
                scores, generate an educational report with Qwen2.5-VL, and export the results.
            </p>
            <div class="badges">
                <span class="badge">MobileNetV3-Small</span>
                <span class="badge">Qwen2.5-VL 3B</span>
                <span class="badge">4 classes</span>
                <span class="badge">Runs on CPU</span>
            </div>
        </div>
        <div class="notice">
            <div>⚠️</div>
            <div><b>Educational prototype only.</b> Model scores are not confirmed disease
            probabilities and must not be used as a clinical diagnosis.</div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

render_stepper(
    has_image=st.session_state["image_bytes"] is not None,
    has_results=bool(st.session_state["mri_results"]),
    has_report=bool(st.session_state["mri_report"]),
)

# =========================================================
# UPLOAD + RESULTS
# =========================================================

left_column, right_column = st.columns([1, 1], gap="large")

with left_column:
    st.markdown(
        section_heading("01", "Upload MRI image", "PNG, JPG or JPEG · 2D slice"),
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose an MRI image",
        type=["png", "jpg", "jpeg"],
        help="Upload a supported 2D image for the experimental classifier.",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        new_image_bytes = uploaded_file.getvalue()

        # Reset results if the user changes the uploaded image.
        if new_image_bytes != st.session_state["image_bytes"]:
            st.session_state["image_bytes"] = new_image_bytes
            st.session_state["mri_results"] = None
            st.session_state["mri_report"] = ""
            st.session_state["analysis_error"] = ""
            st.rerun()

        try:
            preview_image = PILImage.open(
                BytesIO(new_image_bytes)
            ).convert("RGB")

            st.image(
                preview_image,
                caption=f"{uploaded_file.name} · {preview_image.width} × {preview_image.height}px",
                use_container_width=True,
            )

        except Exception as error:
            st.error(f"Could not read this image: {error}")

    analyze_clicked = st.button(
        "Analyze MRI Image",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    )

    if analyze_clicked and uploaded_file is not None:
        try:
            image_for_model = PILImage.open(
                BytesIO(st.session_state["image_bytes"])
            ).convert("RGB")

            with st.spinner("Analyzing the image with MobileNetV3..."):
                st.session_state["mri_results"] = classify_mri(
                    image_for_model
                )

            st.session_state["mri_report"] = ""
            st.session_state["analysis_error"] = ""
            st.rerun()

        except Exception as error:
            st.session_state["analysis_error"] = str(error)
            st.session_state["mri_results"] = None

    if st.session_state["analysis_error"]:
        st.error(
            "Analysis failed: "
            + st.session_state["analysis_error"]
        )


with right_column:
    st.markdown(
        section_heading("02", "Classification results", "Experimental model scores"),
        unsafe_allow_html=True,
    )

    results = st.session_state["mri_results"]

    if results:
        render_results(results)
    else:
        empty_state(
            "🔬",
            "No results yet",
            "Upload an image and select “Analyze MRI Image” to see the classification scores.",
        )


# =========================================================
# REPORT + EXPORT (TABS)
# =========================================================

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

results = st.session_state["mri_results"]
image_bytes = st.session_state["image_bytes"]

tab_report, tab_export = st.tabs(["✦  AI-Assisted Report", "📄  Export"])

with tab_report:
    st.markdown(
        section_heading("03", "AI-assisted report", "Educational summary drafted by Qwen2.5-VL"),
        unsafe_allow_html=True,
    )

    if results and image_bytes:
        st.write(
            "Qwen2.5-VL can draft an educational summary based on "
            "the uploaded image and the classifier scores."
        )

        generate_clicked = st.button(
            "✦ Generate Educational Report",
            use_container_width=True,
        )

        if generate_clicked:
            try:
                with st.spinner(
                    "Generating the report with Qwen2.5-VL..."
                ):
                    st.session_state["mri_report"] = generate_report(
                        image_bytes,
                        results,
                    )

                st.rerun()

            except requests.ConnectionError:
                st.error(
                    "Could not connect to Ollama. Open Ollama and make "
                    "sure the local API is available."
                )

            except requests.Timeout:
                st.error(
                    "The report request timed out. Try again after "
                    "checking that Qwen2.5-VL is running."
                )

            except Exception as error:
                st.error(f"Report generation failed: {error}")

        report_text = st.session_state["mri_report"]

        if report_text:
            st.success("Report generated.")
            st.markdown('<div class="report-box">', unsafe_allow_html=True)
            st.markdown(report_text)
            st.markdown("</div>", unsafe_allow_html=True)

            st.download_button(
                label="Download Report (.txt)",
                data=report_text,
                file_name="neuroscan_mri_report.txt",
                mime="text/plain",
                use_container_width=True,
            )

        else:
            st.caption(
                "Generate the report first. You can then download it as TXT or PDF."
            )

    else:
        empty_state(
            "📝",
            "Report not available yet",
            "Analyze an MRI image first to enable educational report generation.",
        )

with tab_export:
    st.markdown(
        section_heading("04", "Export report", "PDF with image, scores and disclaimer"),
        unsafe_allow_html=True,
    )

    report_text = st.session_state["mri_report"]

    if results and image_bytes:
        st.write(
            "The PDF includes the uploaded image, classifier scores, "
            "any generated Qwen report, and the safety disclaimer."
        )

        try:
            pdf_data = create_pdf_report(
                image_bytes=image_bytes,
                results=results,
                report_text=report_text,
            )

            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_data,
                file_name="neuroscan_mri_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        except Exception as error:
            st.error(f"Could not create the PDF report: {error}")

    else:
        empty_state(
            "📦",
            "Nothing to export yet",
            "Run the classifier first to enable PDF export.",
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    html_block(
        """
        <div class="app-footer">
            <b>NeuroScan AI</b> · Experimental AI coursework project<br>
            For educational and research purposes only. Not a substitute for professional medical interpretation.
        </div>
        """
    ),
    unsafe_allow_html=True,
)