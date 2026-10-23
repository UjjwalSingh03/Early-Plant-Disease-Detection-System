"""
Plant Disease Detection Project — Complete PDF Guide Generator
Uses ReportLab to produce a fully formatted, multi-section PDF documentation.
"""

import os
import json
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, ListFlowable, ListItem, KeepTogether
)
from reportlab.platypus import Frame, PageTemplate, BaseDocTemplate
from reportlab.lib.colors import HexColor

# ── Color Palette ──────────────────────────────────────────────────────────────
DARK_BG       = HexColor("#0F2027")
ACCENT_GREEN  = HexColor("#2ECC71")
ACCENT_BLUE   = HexColor("#3498DB")
ACCENT_ORANGE = HexColor("#E67E22")
ACCENT_RED    = HexColor("#E74C3C")
ACCENT_PURPLE = HexColor("#9B59B6")
LIGHT_GRAY    = HexColor("#ECF0F1")
MID_GRAY      = HexColor("#BDC3C7")
DARK_GRAY     = HexColor("#2C3E50")
TABLE_HEADER  = HexColor("#1A5276")
TABLE_ROW1    = HexColor("#D6EAF8")
TABLE_ROW2    = HexColor("#FDFEFE")
CODE_BG       = HexColor("#1E272E")
CODE_FG       = HexColor("#A8D8A8")
FOLDER_COLOR  = HexColor("#F39C12")
FILE_COLOR    = HexColor("#3498DB")
WHITE         = colors.white
BLACK         = colors.black


OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "Plant_Disease_Project_Complete_Guide.pdf")

# ── Custom Page Template with header/footer ────────────────────────────────────
class ProjectDocTemplate(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        self.page_width, self.page_height = A4
        self.addPageTemplates([
            PageTemplate(
                id="main",
                frames=[Frame(
                    1.5 * cm, 2.0 * cm,
                    self.page_width - 3.0 * cm,
                    self.page_height - 3.5 * cm,
                    id="main_frame"
                )],
                onPage=self._draw_page
            )
        ])

    def _draw_page(self, canvas, doc):
        canvas.saveState()
        w, h = A4
        # Top gradient bar
        canvas.setFillColor(DARK_BG)
        canvas.rect(0, h - 1.5 * cm, w, 1.5 * cm, fill=1, stroke=0)
        canvas.setFillColor(ACCENT_GREEN)
        canvas.rect(0, h - 1.5 * cm, w * 0.4, 1.5 * cm, fill=1, stroke=0)
        canvas.setFillColor(ACCENT_BLUE)
        canvas.rect(w * 0.4, h - 1.5 * cm, w * 0.35, 1.5 * cm, fill=1, stroke=0)
        canvas.setFillColor(DARK_BG)
        canvas.rect(w * 0.75, h - 1.5 * cm, w * 0.25, 1.5 * cm, fill=1, stroke=0)

        # Header text
        canvas.setFont("Helvetica-Bold", 9)
        canvas.setFillColor(WHITE)
        canvas.drawString(1.5 * cm, h - 1.0 * cm, "🌿 Plant Disease Detection — CNN Deep Learning Project")
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(w - 1.5 * cm, h - 1.0 * cm, f"Final Year Capstone Project Guide")

        # Bottom bar
        canvas.setFillColor(DARK_BG)
        canvas.rect(0, 0, w, 1.5 * cm, fill=1, stroke=0)
        canvas.setFillColor(ACCENT_GREEN)
        canvas.rect(0, 0, 0.5 * cm, 1.5 * cm, fill=1, stroke=0)
        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(1.2 * cm, 0.6 * cm, "Phase 1 — Project Documentation & File Reference Guide")
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawRightString(w - 1.5 * cm, 0.6 * cm, f"Page {doc.page}")
        canvas.restoreState()


# ── Style Definitions ──────────────────────────────────────────────────────────
def build_styles():
    base = getSampleStyleSheet()
    styles = {}

    styles["cover_title"] = ParagraphStyle(
        "cover_title", parent=base["Title"],
        fontSize=32, leading=40, textColor=WHITE,
        alignment=TA_CENTER, spaceAfter=12, fontName="Helvetica-Bold"
    )
    styles["cover_sub"] = ParagraphStyle(
        "cover_sub", parent=base["Normal"],
        fontSize=14, leading=20, textColor=ACCENT_GREEN,
        alignment=TA_CENTER, spaceAfter=6, fontName="Helvetica-BoldOblique"
    )
    styles["cover_info"] = ParagraphStyle(
        "cover_info", parent=base["Normal"],
        fontSize=11, leading=16, textColor=MID_GRAY,
        alignment=TA_CENTER, spaceAfter=4, fontName="Helvetica"
    )
    styles["chapter_title"] = ParagraphStyle(
        "chapter_title", parent=base["Heading1"],
        fontSize=20, leading=28, textColor=WHITE,
        spaceBefore=6, spaceAfter=14, fontName="Helvetica-Bold",
        backColor=DARK_BG, borderPad=(8, 8, 8, 8), leftIndent=-10,
    )
    styles["section_title"] = ParagraphStyle(
        "section_title", parent=base["Heading2"],
        fontSize=14, leading=20, textColor=ACCENT_BLUE,
        spaceBefore=12, spaceAfter=6, fontName="Helvetica-Bold",
        borderPad=(0, 0, 2, 0),
    )
    styles["subsection_title"] = ParagraphStyle(
        "subsection_title", parent=base["Heading3"],
        fontSize=11, leading=16, textColor=ACCENT_ORANGE,
        spaceBefore=8, spaceAfter=4, fontName="Helvetica-Bold",
    )
    styles["body"] = ParagraphStyle(
        "body", parent=base["Normal"],
        fontSize=9.5, leading=14.5, textColor=DARK_GRAY,
        alignment=TA_JUSTIFY, spaceAfter=6, fontName="Helvetica"
    )
    styles["body_bold"] = ParagraphStyle(
        "body_bold", parent=base["Normal"],
        fontSize=9.5, leading=14.5, textColor=DARK_GRAY,
        alignment=TA_LEFT, spaceAfter=4, fontName="Helvetica-Bold"
    )
    styles["code"] = ParagraphStyle(
        "code", parent=base["Code"],
        fontSize=8, leading=12, textColor=CODE_FG,
        fontName="Courier", backColor=CODE_BG,
        spaceAfter=4, spaceBefore=4,
        leftIndent=10, rightIndent=10, borderPad=(6, 6, 6, 6)
    )
    styles["file_path"] = ParagraphStyle(
        "file_path", parent=base["Normal"],
        fontSize=9, leading=13, textColor=ACCENT_GREEN,
        fontName="Courier-Bold", spaceAfter=4
    )
    styles["folder_label"] = ParagraphStyle(
        "folder_label", parent=base["Normal"],
        fontSize=12, leading=18, textColor=FOLDER_COLOR,
        fontName="Helvetica-Bold", spaceAfter=4, spaceBefore=10
    )
    styles["note"] = ParagraphStyle(
        "note", parent=base["Normal"],
        fontSize=9, leading=13, textColor=HexColor("#5D6D7E"),
        fontName="Helvetica-Oblique", spaceAfter=4, leftIndent=14
    )
    styles["toc_entry"] = ParagraphStyle(
        "toc_entry", parent=base["Normal"],
        fontSize=10, leading=16, textColor=DARK_GRAY,
        fontName="Helvetica", spaceAfter=2
    )
    styles["toc_chapter"] = ParagraphStyle(
        "toc_chapter", parent=base["Normal"],
        fontSize=11, leading=18, textColor=ACCENT_BLUE,
        fontName="Helvetica-Bold", spaceAfter=3, spaceBefore=6
    )
    styles["badge_green"] = ParagraphStyle(
        "badge_green", parent=base["Normal"],
        fontSize=8, leading=12, textColor=WHITE,
        fontName="Helvetica-Bold", backColor=ACCENT_GREEN,
        borderPad=(4, 2, 4, 2), spaceAfter=2
    )
    styles["callout"] = ParagraphStyle(
        "callout", parent=base["Normal"],
        fontSize=9, leading=14, textColor=HexColor("#154360"),
        fontName="Helvetica", backColor=HexColor("#D6EAF8"),
        borderPad=(8, 6, 8, 6), leftIndent=10, rightIndent=10, spaceAfter=8
    )
    return styles

S = build_styles()


# ── Helper builders ────────────────────────────────────────────────────────────
def ch(title, number=None):
    label = f"{'0'+str(number)+'.  ' if number else ''}{title.upper()}"
    return [
        PageBreak(),
        Spacer(1, 0.3 * cm),
        Paragraph(f"  {label}", S["chapter_title"]),
        HRFlowable(width="100%", thickness=2, color=ACCENT_GREEN, spaceAfter=10),
    ]

def sec(title):
    return [Paragraph(title, S["section_title"])]

def subsec(title):
    return [Paragraph(title, S["subsection_title"])]

def body(text):
    return [Paragraph(text, S["body"])]

def code_block(code_text):
    lines = code_text.strip().split("\n")
    paras = []
    for line in lines:
        escaped = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        paras.append(Paragraph(escaped if escaped else "&nbsp;", S["code"]))
    return paras

def file_header(icon, name, path):
    return [
        Spacer(1, 0.2 * cm),
        Paragraph(f"{icon}  {name}", S["subsection_title"]),
        Paragraph(f"📁 {path}", S["file_path"]),
        HRFlowable(width="100%", thickness=0.5, color=MID_GRAY, spaceAfter=4),
    ]

def folder_header(icon, name):
    return [
        Spacer(1, 0.15 * cm),
        Paragraph(f"{icon}  {name}", S["folder_label"]),
    ]

def callout(text):
    return [Paragraph(text, S["callout"])]

def build_table(header_row, data_rows, col_widths=None, accent=TABLE_HEADER):
    w = A4[0] - 3.0 * cm
    if col_widths is None:
        col_widths = [w / len(header_row)] * len(header_row)

    header = [Paragraph(f"<b>{h}</b>", ParagraphStyle(
        "th", fontSize=8.5, textColor=WHITE, fontName="Helvetica-Bold", leading=12
    )) for h in header_row]

    table_data = [header]
    for i, row in enumerate(data_rows):
        styled = [Paragraph(str(cell), ParagraphStyle(
            "td", fontSize=8, textColor=DARK_GRAY, fontName="Helvetica",
            leading=11, wordWrap='LTR'
        )) for cell in row]
        table_data.append(styled)

    row_colors = []
    for i in range(1, len(table_data)):
        bg = TABLE_ROW1 if i % 2 == 0 else TABLE_ROW2
        row_colors.append(("BACKGROUND", (0, i), (-1, i), bg))

    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), accent),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [TABLE_ROW2, TABLE_ROW1]),
        ("GRID", (0, 0), (-1, -1), 0.5, MID_GRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [TABLE_ROW2, TABLE_ROW1]),
    ] + row_colors)

    return [Table(table_data, colWidths=col_widths, style=style, repeatRows=1), Spacer(1, 0.3 * cm)]


# ══════════════════════════════════════════════════════════════════════════════
#  DOCUMENT CONTENT BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def build_document():
    story = []
    w = A4[0] - 3.0 * cm

    # ── COVER PAGE ──────────────────────────────────────────────────────────
    story.append(Spacer(1, 3.5 * cm))
    story.append(Paragraph("🌿", ParagraphStyle("emoji", fontSize=60, alignment=TA_CENTER, leading=72)))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Automated Plant Disease Detection", S["cover_title"]))
    story.append(Paragraph("Using Deep Convolutional Neural Networks", S["cover_title"]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width="60%", thickness=3, color=ACCENT_GREEN, hAlign="CENTER", spaceAfter=14))
    story.append(Paragraph("COMPLETE PROJECT FILE-BY-FILE REFERENCE GUIDE", S["cover_sub"]))
    story.append(Spacer(1, 0.6 * cm))

    cover_info = [
        ["Phase", "Phase 1 — Project Documentation & Architecture Baseline"],
        ["Dataset", "PlantVillage Dataset · 54,305 Images · 38 Classes · 14 Crops"],
        ["Framework", "TensorFlow 2.15 · Keras · Streamlit 1.30 · Python 3.10+"],
        ["Architecture", "Custom Sequential CNN · 47,805,158 Parameters · ~182 MB"],
        ["Baseline Accuracy", "88.28% Validation Accuracy (5 Epochs)"],
        ["Repository", "siddhardhan23 / plant-disease-prediction-cnn-deep-leanring-project"],
    ]
    cover_table = Table(
        [[Paragraph(f"<b>{r[0]}</b>", ParagraphStyle("ci_k", fontSize=9, textColor=ACCENT_GREEN, fontName="Helvetica-Bold", leading=13)),
          Paragraph(r[1], ParagraphStyle("ci_v", fontSize=9, textColor=MID_GRAY, fontName="Helvetica", leading=13))]
         for r in cover_info],
        colWidths=[4.2 * cm, w - 4.2 * cm],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), DARK_BG),
            ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#2C3E50")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ])
    )
    story.append(cover_table)
    story.append(Spacer(1, 0.6 * cm))
    story.append(HRFlowable(width="60%", thickness=1, color=ACCENT_BLUE, hAlign="CENTER", spaceAfter=8))
    story.append(Paragraph("Final Year Engineering Capstone Project", S["cover_info"]))
    story.append(Paragraph("Document Status: Phase 1 Complete · Ready for First Representation", S["cover_info"]))

    # ── TABLE OF CONTENTS ───────────────────────────────────────────────────
    story += ch("Table of Contents")
    toc_data = [
        ("01", "Project Overview & Structure",              "A complete map of the project folder hierarchy"),
        ("02", "Root Directory Files",                      "PROJECT_PHASE_TRACKER.md and directory overview"),
        ("03", "Folder: app/",                             "Streamlit app, Dockerfile, config, class labels"),
        ("  03.1", "  app/main.py",                        "Main Streamlit application entry point"),
        ("  03.2", "  app/class_indices.json",             "38-class label lookup dictionary"),
        ("  03.3", "  app/Dockerfile",                     "Docker container definition"),
        ("  03.4", "  app/requirements.txt",               "Python dependency pinning"),
        ("  03.5", "  app/config.toml",                    "Streamlit server configuration"),
        ("  03.6", "  app/credentials.toml",               "Streamlit auth credentials"),
        ("  03.7", "  app/trained_model/trained_model_link.txt", "Google Drive model download link"),
        ("04", "Folder: notebooks/",                       "Model training Jupyter notebook"),
        ("  04.1", "  Notebook: Cell-by-Cell Walkthrough", "Full 43-cell annotation & explanation"),
        ("05", "Folder: test_samples/",                    "Sample leaf test images for demonstration"),
        ("06", "Folder: Phase_1_Representation/",         "All Phase 1 academic report documents"),
        ("  06.1", "  PHASE_1_PROJECT_REPORT.md",         "Formal capstone report (synopsis to results)"),
        ("  06.2", "  PHASE_1_PRESENTATION_SLIDES.md",    "10-slide deck + viva Q&A guide"),
        ("  06.3", "  tables/ — 5 Reference Tables",      "Dataset, CNN specs, training metrics, tools, comparison"),
        ("07", "Dataset Reference — 38 Classes Full Table","Complete crop × disease mapping with pathogen info"),
        ("08", "CNN Architecture Deep Dive",               "Layer shapes, parameters, math formulas"),
        ("09", "Training Results & Performance Metrics",   "Epoch log, convergence, gap analysis"),
        ("10", "Phase Roadmap — What Comes Next",          "Phase 2 to Phase 4 planned deliverables"),
    ]
    toc_rows = [[Paragraph(f"<b>{n}</b>", ParagraphStyle("tn", fontSize=9, textColor=ACCENT_BLUE, fontName="Helvetica-Bold", leading=12)),
                 Paragraph(title, ParagraphStyle("tt", fontSize=9, textColor=DARK_GRAY, fontName="Helvetica", leading=12)),
                 Paragraph(desc, ParagraphStyle("td", fontSize=8, textColor=HexColor("#7F8C8D"), fontName="Helvetica-Oblique", leading=11))]
                for n, title, desc in toc_data]
    toc_table = Table(toc_rows, colWidths=[1.5*cm, 6.5*cm, w-8.0*cm],
                      style=TableStyle([
                          ("GRID", (0,0), (-1,-1), 0.3, LIGHT_GRAY),
                          ("ROWBACKGROUNDS", (0,0), (-1,-1), [TABLE_ROW2, TABLE_ROW1]),
                          ("TOPPADDING", (0,0), (-1,-1), 5),
                          ("BOTTOMPADDING", (0,0), (-1,-1), 5),
                          ("LEFTPADDING", (0,0), (-1,-1), 8),
                      ]))
    story.append(toc_table)

    # ════════════════════════════════════════════════════════════════════════
    # CHAPTER 01 — PROJECT OVERVIEW & FOLDER STRUCTURE
    # ════════════════════════════════════════════════════════════════════════
    story += ch("Project Overview & Directory Structure", 1)
    story += body(
        "This guide provides a comprehensive, annotated walkthrough of every file and folder in the "
        "<b>Automated Plant Disease Detection CNN Project</b>. Each file is explained in context: "
        "its purpose, how it connects to the overall system pipeline, code walkthrough, and "
        "configuration details. This document is intended to support your Final Year Project review, "
        "guide viva preparation, and serve as a living technical reference through all four project phases."
    )
    story.append(Spacer(1, 0.3 * cm))

    tree_text = """
Disease/                               ← Project Root Workspace
│
├── PROJECT_PHASE_TRACKER.md           ← Multi-phase milestone tracker
│
├── Phase_1_Representation/            ← All Phase 1 academic documents
│   ├── PHASE_1_PROJECT_REPORT.md      ← Formal capstone report (synopsis → results)
│   ├── PHASE_1_PRESENTATION_SLIDES.md ← 10-slide showcase deck + viva Q&A
│   └── tables/                        ← 5 reference data & architecture tables
│       ├── 01_dataset_taxonomy_38_classes.md
│       ├── 02_cnn_architecture_specs.md
│       ├── 03_baseline_training_metrics.md
│       ├── 04_system_requirements_matrix.md
│       └── 05_comparative_study_table.md
│
├── app/                               ← Streamlit Web Application
│   ├── main.py                        ← Main inference & UI app (entry point)
│   ├── class_indices.json             ← 38-class label dictionary
│   ├── Dockerfile                     ← Docker container definition
│   ├── requirements.txt               ← Python dependency pinning
│   ├── config.toml                    ← Streamlit server configuration
│   ├── credentials.toml               ← Streamlit authentication file
│   └── trained_model/
│       └── trained_model_link.txt     ← Google Drive link to .h5 model weights
│
├── notebooks/                         ← Jupyter Notebook (Model Training)
│   └── Plant_Disease_Prediction_CNN_Image_Classifier.ipynb
│
├── test_samples/                      ← Sample leaf images for live demo
│   ├── test_apple_black_rot.JPG
│   ├── test_blueberry_healthy.jpg
│   └── test_potato_early_blight.jpg
│
└── docs/                              ← This PDF guide lives here
    └── Plant_Disease_Project_Complete_Guide.pdf
"""
    story += code_block(tree_text)
    story += callout(
        "📌  How to read this guide: Each chapter covers one folder or file in depth. "
        "Start with Chapter 03 (app/) for running the app, Chapter 04 (notebooks/) for "
        "training the model, and Chapter 06 (Phase_1_Representation/) for academic documents."
    )

    # ════════════════════════════════════════════════════════════════════════
    # CHAPTER 02 — ROOT DIRECTORY FILES
    # ════════════════════════════════════════════════════════════════════════
    story += ch("Root Directory Files", 2)

    story += file_header("📋", "PROJECT_PHASE_TRACKER.md", "Disease/PROJECT_PHASE_TRACKER.md")
    story += body(
        "This is the <b>master project state file</b>. It acts as a living checkpoint document "
        "that tracks the progress, completion status, and deliverables for each phase of the "
        "capstone project. Every time you complete a phase and ask to proceed to the next phase, "
        "this file gets updated to reflect the new state."
    )
    story += sec("What It Contains")
    story += body("• <b>Current Status Banner</b>: Shows which phase is active right now.")
    story += body("• <b>Phase 1 Checklist</b>: Lists all generated files and documents with links.")
    story += body("• <b>Phase 2–4 Planned Tasks</b>: Detailed roadmap of upcoming model optimizations, transfer learning benchmarks, mobile export, and final thesis.")
    story += body("• <b>Resume Instruction</b>: Tells you exactly what to say to continue building the project (e.g., 'Start Phase 2').")
    story += callout(
        "💡  Think of PROJECT_PHASE_TRACKER.md as your project's 'save game' file. "
        "It captures the exact state of work so you never lose progress between sessions."
    )

    story += sec("Phase Overview Table")
    story += build_table(
        ["Phase", "Title", "Status", "Key Deliverables"],
        [
            ["Phase 1", "Architecture Baseline & First Representation", "✅ COMPLETED", "Dataset tables, CNN specs, baseline 88.28% accuracy, Streamlit prototype, presentation slides"],
            ["Phase 2", "Optimization & Regularization", "⏳ QUEUED", "Data augmentation, Dropout, Batch Norm, >94% accuracy, Confusion Matrix"],
            ["Phase 3", "Transfer Learning & UI Upgrade", "📅 PLANNED", "MobileNetV2 benchmark, treatment recommendations, TFLite export"],
            ["Phase 4", "Field Validation & Final Thesis", "📅 PLANNED", "Real-world testing, IEEE thesis report, viva defense"],
        ],
        col_widths=[1.5*cm, 5*cm, 2.5*cm, w-9.0*cm]
    )

    # ════════════════════════════════════════════════════════════════════════
    # CHAPTER 03 — app/ FOLDER
    # ════════════════════════════════════════════════════════════════════════
    story += ch("Folder: app/ — The Web Application", 3)
    story += folder_header("📂", "app/")
    story += body(
        "The <b>app/</b> directory contains the complete Streamlit-based web application that "
        "provides real-time plant disease diagnosis. This is the production-ready user-facing "
        "component of the project. It loads the trained CNN model, accepts a user-uploaded leaf "
        "image, preprocesses it, and renders the predicted disease classification result — "
        "all within a clean browser-based interface."
    )

    # 03.1 — main.py
    story += file_header("🐍", "main.py", "Disease/app/main.py")
    story += body(
        "This is the <b>primary entry point</b> of the entire Streamlit application. When you run "
        "<code>streamlit run main.py</code> (or the Docker container starts), this is the script "
        "that executes. It handles model loading, image preprocessing, prediction, and the complete "
        "user interface rendering."
    )
    story += subsec("Complete Annotated Code")
    main_code = """\
import os
import json
from PIL import Image           # Pillow: image loading & resizing
import numpy as np              # NumPy: array operations on pixel data
import tensorflow as tf         # TensorFlow: loading the trained .h5 model
import streamlit as st          # Streamlit: web UI framework

# --- Path Setup ---
# Dynamically resolve the directory where main.py lives.
# This ensures the app works correctly inside Docker too.
working_dir = os.path.dirname(os.path.abspath(__file__))

# Build the absolute path to the model weights file.
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"

# --- Model Loading ---
# Load the pre-trained Keras CNN model from disk.
# This runs ONCE at startup (cached by Streamlit session state).
model = tf.keras.models.load_model(model_path)

# --- Class Dictionary Loading ---
# Load the 38-class index → label name mapping from JSON.
# E.g., {"0": "Apple___Apple_scab", "1": "Apple___Black_rot", ...}
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# ------- FUNCTION: load_and_preprocess_image -------
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)         # Open image via Pillow
    img = img.resize(target_size)        # Resize to 224×224 pixels
    img_array = np.array(img)            # Convert PIL Image → NumPy array (H,W,C)
    img_array = np.expand_dims(          # Add batch dimension → (1,H,W,C)
        img_array, axis=0)
    img_array = img_array.astype(        # Normalize pixel values: [0,255] → [0.0,1.0]
        'float32') / 255.
    return img_array

# ------- FUNCTION: predict_image_class -------
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)        # Forward pass → (1,38) softmax
    predicted_class_index = np.argmax(predictions, axis=1)[0]  # Index of max probability
    predicted_class_name  = class_indices[str(predicted_class_index)]
    return predicted_class_name

# ------- STREAMLIT UI -------
st.title('Plant Disease Classifier')   # Page heading

uploaded_image = st.file_uploader(     # Drag-and-drop file upload widget
    "Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)         # Two-column side-by-side layout

    with col1:
        resized_img = image.resize((150, 150))  # Thumbnail preview
        st.image(resized_img)

    with col2:
        if st.button('Classify'):       # Trigger prediction on button click
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')  # Show result
"""
    story += code_block(main_code)

    story += subsec("Step-by-Step Data Flow")
    story += build_table(
        ["Step", "Operation", "Input → Output", "Key Function / Library"],
        [
            ["1", "Image Upload", "Browser file → PIL Image object", "st.file_uploader() + PIL.Image.open()"],
            ["2", "Image Resize", "Variable resolution → 224×224 px", "img.resize((224,224))"],
            ["3", "Array Conversion", "PIL Image → NumPy array (H,W,C)", "np.array(img)"],
            ["4", "Batch Dimension", "Shape (224,224,3) → (1,224,224,3)", "np.expand_dims(axis=0)"],
            ["5", "Normalization", "Pixel range [0,255] → [0.0,1.0]", "img_array / 255."],
            ["6", "CNN Forward Pass", "Tensor (1,224,224,3) → Softmax (1,38)", "model.predict()"],
            ["7", "Class Decoding", "Argmax index (0–37) → Label string", "class_indices[str(idx)]"],
            ["8", "Display Result", "Label string → st.success() banner", "Streamlit UI"],
        ],
        col_widths=[0.8*cm, 3.5*cm, 4.5*cm, w-8.8*cm]
    )

    # 03.2 — class_indices.json
    story += file_header("🔖", "class_indices.json", "Disease/app/class_indices.json")
    story += body(
        "This JSON file is the <b>label decoder dictionary</b>. It maps each integer class index "
        "(0 to 37) to its human-readable plant disease label. The CNN model outputs a raw numeric "
        "index — this file is what converts that number into a meaningful diagnosis name like "
        "<b>'Tomato___Early_blight'</b> or <b>'Apple___healthy'</b>."
    )
    story += subsec("File Format Sample")
    story += code_block("""\
{
  "0":  "Apple___Apple_scab",
  "1":  "Apple___Black_rot",
  "2":  "Apple___Cedar_apple_rust",
  "3":  "Apple___healthy",
  "4":  "Blueberry___healthy",
  "5":  "Cherry_(including_sour)___Powdery_mildew",
  ...
  "37": "Tomato___healthy"
}
""")
    story += body(
        "The naming convention is: <b>CropName___ConditionName</b>. The triple underscore "
        "(<code>___</code>) is used as a delimiter to separate the plant species from the "
        "specific condition or disease name. This is consistent with the PlantVillage dataset "
        "folder naming scheme."
    )

    # Full class table
    story += subsec("Complete 38-Class Lookup Table")
    class_rows = [
        ["0", "Apple", "Apple Scab", "Fungal"], ["1", "Apple", "Black Rot", "Fungal"],
        ["2", "Apple", "Cedar Apple Rust", "Fungal"], ["3", "Apple", "Healthy", "—"],
        ["4", "Blueberry", "Healthy", "—"], ["5", "Cherry", "Powdery Mildew", "Fungal"],
        ["6", "Cherry", "Healthy", "—"], ["7", "Corn", "Cercospora / Gray Leaf Spot", "Fungal"],
        ["8", "Corn", "Common Rust", "Fungal"], ["9", "Corn", "Northern Leaf Blight", "Fungal"],
        ["10", "Corn", "Healthy", "—"], ["11", "Grape", "Black Rot", "Fungal"],
        ["12", "Grape", "Esca (Black Measles)", "Fungal Complex"], ["13", "Grape", "Leaf Blight", "Fungal"],
        ["14", "Grape", "Healthy", "—"], ["15", "Orange", "Huanglongbing (Citrus Greening)", "Bacterial"],
        ["16", "Peach", "Bacterial Spot", "Bacterial"], ["17", "Peach", "Healthy", "—"],
        ["18", "Pepper Bell", "Bacterial Spot", "Bacterial"], ["19", "Pepper Bell", "Healthy", "—"],
        ["20", "Potato", "Early Blight", "Fungal"], ["21", "Potato", "Late Blight", "Oomycete"],
        ["22", "Potato", "Healthy", "—"], ["23", "Raspberry", "Healthy", "—"],
        ["24", "Soybean", "Healthy", "—"], ["25", "Squash", "Powdery Mildew", "Fungal"],
        ["26", "Strawberry", "Leaf Scorch", "Fungal"], ["27", "Strawberry", "Healthy", "—"],
        ["28", "Tomato", "Bacterial Spot", "Bacterial"], ["29", "Tomato", "Early Blight", "Fungal"],
        ["30", "Tomato", "Late Blight", "Oomycete"], ["31", "Tomato", "Leaf Mold", "Fungal"],
        ["32", "Tomato", "Septoria Leaf Spot", "Fungal"], ["33", "Tomato", "Spider Mite Damage", "Pest"],
        ["34", "Tomato", "Target Spot", "Fungal"], ["35", "Tomato", "Yellow Leaf Curl Virus", "Viral"],
        ["36", "Tomato", "Mosaic Virus", "Viral"], ["37", "Tomato", "Healthy", "—"],
    ]
    story += build_table(
        ["ID", "Crop", "Condition / Disease", "Pathogen Type"],
        class_rows,
        col_widths=[1.0*cm, 3.0*cm, 6.0*cm, w-10.0*cm]
    )

    # 03.3 — Dockerfile
    story += file_header("🐳", "Dockerfile", "Disease/app/Dockerfile")
    story += body(
        "The <b>Dockerfile</b> defines the complete containerized runtime environment for the "
        "Streamlit application. This allows the app to run identically on any system — cloud server, "
        "local machine, or remote cluster — regardless of the host operating system or Python "
        "environment, eliminating the classic 'works on my machine' problem."
    )
    story += code_block("""\
# Base image: Official Python 3.10 slim (minimal footprint)
FROM python:3.10-slim

# Copy ALL files from the build context (app/ directory) into /app inside container
COPY . /app

# Set working directory for all subsequent commands
WORKDIR /app

# Install all Python dependencies from requirements.txt
RUN pip install -r requirements.txt

# Expose port 80 (HTTP standard) — Streamlit will listen here
EXPOSE 80

# Create Streamlit config directory and copy configuration files
RUN mkdir ~/.streamlit
RUN cp config.toml ~/.streamlit/config.toml
RUN cp credentials.toml ~/.streamlit/credentials.toml

# Container startup: run Streamlit server with main.py
ENTRYPOINT ["streamlit", "run"]
CMD ["main.py"]
""")

    story += subsec("Docker Build & Run Commands")
    story += code_block("""\
# Build the Docker image (run from inside the app/ directory)
docker build -t plant-disease-app .

# Run the container, mapping host port 8501 → container port 80
docker run -p 8501:80 plant-disease-app

# Then open browser at: http://localhost:8501
""")

    # 03.4 — requirements.txt
    story += file_header("📦", "requirements.txt", "Disease/app/requirements.txt")
    story += body(
        "This file pins the exact Python package versions required to run the application. "
        "Using exact version pinning (e.g., <code>tensorflow==2.15.0.post1</code>) ensures "
        "reproducible behavior across development, testing, and deployment environments."
    )
    story += code_block("""\
numpy==1.26.3           # Array operations, tensor manipulation
streamlit==1.30.0       # Web UI framework
tensorflow==2.15.0.post1  # Deep learning model loading and inference
""")
    story += build_table(
        ["Package", "Version", "Purpose"],
        [
            ["numpy", "1.26.3", "Multi-dimensional array operations, image pixel manipulation, argmax for class decoding"],
            ["streamlit", "1.30.0", "Browser-based interactive web app framework; handles UI, file upload, and display"],
            ["tensorflow", "2.15.0.post1", "Loads the trained CNN model (.h5 format) and executes the forward pass inference"],
        ],
        col_widths=[3.0*cm, 2.5*cm, w-5.5*cm]
    )

    # 03.5 — config.toml
    story += file_header("⚙️", "config.toml", "Disease/app/config.toml")
    story += body(
        "The Streamlit configuration file controls server behavior, logging, browser connectivity, "
        "and port settings. Critically, it sets <b>headless = true</b> and <b>port = 80</b> to "
        "make the app work correctly inside Docker containers and cloud deployments."
    )
    story += code_block("""\
[global]
showWarningOnDirectExecution = true   # Warn if run via 'python main.py' instead of streamlit

[logger]
level = "debug"                        # Log all debug messages (verbose for development)

[runner]
magicEnabled = true                    # Allow writing variables directly (magic mode)

[server]
headless = true                        # Don't try to open browser window (needed for Docker/Cloud)
port = 80                              # Listen on port 80 (HTTP standard)
enableCORS = false                     # Disable CORS restrictions (allow cross-origin in Docker)
runOnSave = false                      # Don't auto-reload on file save (stability)

[browser]
serverAddress = "0.0.0.0"             # Accept connections from any IP (required for Docker)
serverPort = 80                        # Browser connects to port 80
gatherUsageStats = true               # Anonymous usage telemetry to Streamlit
""")

    # 03.6 — credentials.toml
    story += file_header("🔐", "credentials.toml", "Disease/app/credentials.toml")
    story += body(
        "This minimal file satisfies Streamlit's requirement for a credentials configuration "
        "file. In the current deployment (no user authentication), it simply contains an empty "
        "email field. If you add Streamlit sharing or user login in Phase 3, this file would "
        "be updated with auth credentials."
    )
    story += code_block("""\
[general]
email=""    # No email authentication configured for Phase 1 deployment
""")

    # 03.7 — trained_model_link.txt
    story += file_header("🔗", "trained_model_link.txt", "Disease/app/trained_model/trained_model_link.txt")
    story += body(
        "Since GitHub has a file size limit (~100 MB), the trained model file "
        "<code>plant_disease_prediction_model.h5</code> (~182 MB) cannot be committed directly. "
        "This text file contains the Google Drive download link for the model weights."
    )
    story += code_block("""\
trained plant disease prediction model link:
https://drive.google.com/file/d/1rKh-IElSdHTqax7XdfSdZTn-r8T_qWPf/view?usp=drive_link
""")
    story += callout(
        "📥  To run the app: Download the model file from the link above and place it at:\n"
        "app/trained_model/plant_disease_prediction_model.h5\n"
        "Then run: streamlit run app/main.py"
    )

    # ════════════════════════════════════════════════════════════════════════
    # CHAPTER 04 — notebooks/
    # ════════════════════════════════════════════════════════════════════════
    story += ch("Folder: notebooks/ — Model Training Notebook", 4)
    story += folder_header("📂", "notebooks/")
    story += body(
        "The notebooks folder contains the Jupyter notebook used to train the CNN model from scratch. "
        "This notebook was designed to run on <b>Google Colab</b> (free GPU/TPU environment) "
        "and covers the complete end-to-end ML pipeline: data curation from Kaggle, preprocessing, "
        "model architecture definition, training, evaluation, and model export."
    )

    story += file_header("📓", "Plant_Disease_Prediction_CNN_Image_Classifier.ipynb",
                          "Disease/notebooks/Plant_Disease_Prediction_CNN_Image_Classifier.ipynb")
    story += body(
        "This notebook contains <b>43 cells</b> covering the complete deep learning training pipeline. "
        "Below is a section-by-section annotated walkthrough of every major cell group."
    )

    sections = [
        ("SECTION 1 — Reproducibility Seeding (Cells 0–1)", [
            ("Purpose", "Set fixed random seeds across Python, NumPy, and TensorFlow to ensure every training run produces identical results."),
            ("Why it matters", "Without seeding, weight initializations are random — results would differ between runs, making it impossible to reproduce or compare experiments."),
        ], """\
import random
random.seed(0)         # Python random module seed
import numpy as np
np.random.seed(0)      # NumPy random seed
import tensorflow as tf
tf.random.set_seed(0)  # TensorFlow global random seed
"""),
        ("SECTION 2 — Dependency Imports (Cells 2–3)", [
            ("Purpose", "Import all required libraries in one place at the top of the notebook."),
            ("Key imports", "os, json, ZipFile (file ops); PIL.Image (image loading); numpy (arrays); matplotlib (visualization); Keras ImageDataGenerator, layers, models (deep learning pipeline)."),
        ], """\
import os, json
from zipfile import ZipFile
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
"""),
        ("SECTION 3 — Kaggle Dataset Download (Cells 4–12)", [
            ("Purpose", "Authenticate with the Kaggle API and programmatically download the PlantVillage dataset (~2.5 GB compressed)."),
            ("How it works", "Install the kaggle CLI, set KAGGLE_USERNAME and KAGGLE_KEY environment variables from the uploaded kaggle.json file, then run the dataset download command."),
        ], """\
!pip install kaggle
kaggle_credentials = json.load(open("kaggle.json"))
os.environ['KAGGLE_USERNAME'] = kaggle_credentials["username"]
os.environ['KAGGLE_KEY'] = kaggle_credentials["key"]
!kaggle datasets download -d abdallahalidev/plantvillage-dataset
# Then unzip:
with ZipFile("plantvillage-dataset.zip", 'r') as zip_ref:
    zip_ref.extractall()
"""),
        ("SECTION 4 — Image Preprocessing (Cells 15–19)", [
            ("Purpose", "Standardize all images to uniform dimensions (224×224) and normalize pixel values to [0, 1] before feeding them into the CNN."),
            ("ImageDataGenerator", "Keras ImageDataGenerator handles batched loading from disk, rescaling (1/255), and train/validation splitting (80:20)."),
        ], """\
# Image Parameters
img_size   = 224       # Target height and width in pixels
batch_size = 32        # Number of images processed per training step

# Data generator: rescale pixels from [0,255] to [0.0,1.0]
data_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2    # Reserve 20% for validation
)

# Training set: 43,456 images from 38 class folders
train_generator = data_gen.flow_from_directory(
    base_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    subset='training',
    class_mode='categorical'
)
# Output: "Found 43456 images belonging to 38 classes."

# Validation set: 10,849 images
validation_generator = data_gen.flow_from_directory(
    base_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    subset='validation',
    class_mode='categorical'
)
# Output: "Found 10849 images belonging to 38 classes."
"""),
        ("SECTION 5 — CNN Model Architecture (Cells 24–27)", [
            ("Purpose", "Define the Sequential CNN architecture with 2 convolutional blocks followed by dense classification layers."),
            ("Architecture decisions", "Conv2D → MaxPooling → Conv2D → MaxPooling → Flatten → Dense(256) → Dense(38,Softmax)"),
        ], """\
model = models.Sequential()

# Block 1: Local edge and texture detection
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(224, 224, 3)))
# Output: (None, 222, 222, 32)  — 896 parameters
model.add(layers.MaxPooling2D(2, 2))
# Output: (None, 111, 111, 32)  — 0 parameters

# Block 2: Complex lesion and pattern detection
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# Output: (None, 109, 109, 64)  — 18,496 parameters
model.add(layers.MaxPooling2D(2, 2))
# Output: (None, 54, 54, 64)    — 0 parameters

# Classification head
model.add(layers.Flatten())
# Output: (None, 186624)         — 0 parameters

model.add(layers.Dense(256, activation='relu'))
# Output: (None, 256)            — 47,776,000 parameters

model.add(layers.Dense(38, activation='softmax'))
# Output: (None, 38)             — 9,766 parameters

# Total: 47,805,158 parameters (~182 MB)

# Compile: Adam optimizer + Categorical Cross-Entropy loss
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
"""),
        ("SECTION 6 — Model Training (Cell 29)", [
            ("Purpose", "Execute the forward/backward pass training loop over 5 epochs. Each epoch processes all 43,456 training images in batches of 32."),
            ("Performance", "Steps per epoch: 1,358  |  Validation steps: 339  |  Time per epoch: ~104–109 seconds on GPU"),
        ], """\
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,  # 1358
    epochs=5,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size  # 339
)
# Results:
# Epoch 1: loss=0.9791, acc=73.28%, val_acc=84.65%
# Epoch 2: loss=0.2812, acc=91.10%, val_acc=86.55%
# Epoch 3: loss=0.1362, acc=95.53%, val_acc=88.63%
# Epoch 4: loss=0.0891, acc=97.08%, val_acc=87.15%
# Epoch 5: loss=0.0761, acc=97.60%, val_acc=88.28%
"""),
        ("SECTION 7 — Evaluation & Prediction System (Cells 30–38)", [
            ("Purpose", "Evaluate on the full validation set, visualize training curves, build the inference prediction function, and test on sample images."),
            ("Final accuracy", "val_accuracy = 88.28% (339 batches evaluated on validation set)"),
        ], """\
# Evaluate model on full validation set
val_loss, val_accuracy = model.evaluate(validation_generator,
    steps=validation_generator.samples // batch_size)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
# Output: "Validation Accuracy: 88.28%"

# Create class_indices mapping: int index → label string
class_indices = {v: k for k, v in train_generator.class_indices.items()}

# Prediction function
def predict_image_class(model, image_path, class_indices):
    img = load_and_preprocess_image(image_path)
    preds = model.predict(img)
    idx = np.argmax(preds, axis=1)[0]
    return class_indices[idx]

# Save class indices as JSON for the app
json.dump(class_indices, open('class_indices.json', 'w'))
"""),
        ("SECTION 8 — Save Model Weights (Cells 39–41)", [
            ("Purpose", "Export the trained model weights in Keras HDF5 format (.h5) to Google Drive or local disk for deployment."),
            ("File size", "plant_disease_prediction_model.h5 is approximately 182 MB — too large for GitHub direct commit."),
        ], """\
# Save to Google Drive (Google Colab)
model.save('drive/MyDrive/Youtube/trained_models/plant_disease_prediction_model.h5')

# OR save locally
model.save('plant_disease_prediction_model.h5')
"""),
    ]

    for sec_title, notes, code in sections:
        story += sec(sec_title)
        for label, text in notes:
            story += body(f"<b>{label}:</b> {text}")
        story += code_block(code)
        story.append(Spacer(1, 0.2 * cm))

    # ════════════════════════════════════════════════════════════════════════
    # CHAPTER 05 — test_samples/
    # ════════════════════════════════════════════════════════════════════════
    story += ch("Folder: test_samples/ — Demonstration Images", 5)
    story += folder_header("📂", "test_samples/")
    story += body(
        "The test_samples folder contains <b>3 real leaf images</b> for use during live "
        "demonstration and presentation. These are actual photographs from the PlantVillage "
        "dataset and represent three distinct prediction scenarios your CNN can handle."
    )
    story += build_table(
        ["File Name", "Plant", "Condition", "Expected Prediction", "Use Case"],
        [
            ["test_apple_black_rot.JPG", "Apple (Malus domestica)", "Black Rot — Fungal Disease", "Apple___Black_rot  (Class 1)", "Demonstrates a diseased leaf diagnosis — dark necrotic ring lesions on leaf surface"],
            ["test_blueberry_healthy.jpg", "Blueberry (Vaccinium)", "Healthy Leaf", "Blueberry___healthy  (Class 4)", "Demonstrates healthy leaf classification — no disease markers detected"],
            ["test_potato_early_blight.jpg", "Potato (Solanum tuberosum)", "Early Blight — Fungal", "Potato___Early_blight  (Class 20)", "Demonstrates early-stage fungal disease detection — concentric ring patterns"],
        ],
        col_widths=[4.0*cm, 3.0*cm, 2.8*cm, 3.5*cm, w-13.3*cm]
    )
    story += callout(
        "🎯  During your viva demonstration, use all 3 images to show the model correctly:\n"
        "1. Identifies a disease (Apple Black Rot)\n"
        "2. Identifies a healthy plant (Blueberry Healthy)\n"
        "3. Detects an early-stage infection (Potato Early Blight)\n"
        "This covers 3 different crop types and 3 different prediction scenarios."
    )

    # ════════════════════════════════════════════════════════════════════════
    # CHAPTER 06 — Phase_1_Representation/
    # ════════════════════════════════════════════════════════════════════════
    story += ch("Folder: Phase_1_Representation/ — Academic Documents", 6)
    story += folder_header("📂", "Phase_1_Representation/")
    story += body(
        "This folder holds every academic document prepared for your Phase 1 capstone representation. "
        "These files are structured to directly map to standard final year project evaluation requirements: "
        "synopsis submission, presentation slides, reference tables, and methodology documentation."
    )

    story += file_header("📄", "PHASE_1_PROJECT_REPORT.md", "Disease/Phase_1_Representation/PHASE_1_PROJECT_REPORT.md")
    story += body("The main formal project document. Its sections correspond directly to a standard capstone synopsis or review report format.")
    story += build_table(
        ["Report Section", "Content Summary"],
        [
            ["Abstract / Executive Summary", "Problem overview, dataset description, methodology, baseline results, and project significance in under 300 words"],
            ["1. Introduction", "Agricultural problem context, motivation, limitations of traditional scouting, and proposed AI solution"],
            ["2. Objectives & Scope", "14 crops, 38 classes, deep CNN approach, exclusions (no live video streaming in Phase 1)"],
            ["3. Dataset Engineering", "PlantVillage dataset: 54,305 images, 80/20 split, 224×224 standardization, pixel normalization"],
            ["4. Architecture & Methodology", "Sequential CNN design with Conv2D blocks, MaxPooling, Dense layers, Softmax output, Adam optimizer"],
            ["5. Experimental Results", "5-epoch training table, convergence analysis, 88.28% baseline, generalization gap discovery"],
            ["6. System Implementation", "app/main.py walkthrough, class_indices.json decoder, Dockerfile deployment notes"],
            ["7. Comparative Study", "Manual scouting vs. Classical ML (SVM) vs. Deep CNN — speed, accuracy, cost, scalability"],
            ["8. Phase Roadmap", "Gantt-style Phase 1–4 milestone planning from Oct 2026 to Jan 2027"],
            ["9. Sign-Off Checklist", "Phase 1 faculty review evaluation criteria with verification status"],
            ["10. References", "5 academic citations: PlantVillage paper, Mohanty et al., LeCun et al., Adam optimizer paper, Goodfellow textbook"],
        ],
        col_widths=[5.0*cm, w-5.0*cm]
    )

    story += file_header("🎤", "PHASE_1_PRESENTATION_SLIDES.md", "Disease/Phase_1_Representation/PHASE_1_PRESENTATION_SLIDES.md")
    story += body("A 10-slide outline with speaking points for your first review presentation, plus anticipated viva questions with model answers.")
    story += build_table(
        ["Slide #", "Title", "Key Points to Cover"],
        [
            ["Slide 1", "Title & Credentials", "Project title, domain (AI/CV/Agriculture), Phase 1 stage"],
            ["Slide 2", "Problem Statement", "20–40% annual crop loss, manual scouting failures, need for automation"],
            ["Slide 3", "Objectives & Scope", "14 crops, 38 classes, <300ms inference, browser-accessible tool"],
            ["Slide 4", "Dataset Taxonomy", "54,305 images, 80/20 split, 224×224, PlantVillage benchmark"],
            ["Slide 5", "CNN Architecture", "2 Conv blocks, MaxPooling, Flatten, Dense(256), Softmax(38)"],
            ["Slide 6", "Math Formulation", "ReLU, Softmax, Categorical Cross-Entropy, Adam optimizer"],
            ["Slide 7", "Results", "Epoch table, 88.28% val accuracy, overfitting gap analysis"],
            ["Slide 8", "Web App Demo", "Streamlit UI, file upload, two-column layout, live diagnosis"],
            ["Slide 9", "Project Roadmap", "Phase 1 done → Phase 2 (Dropout/Augmentation) → Phase 3 (TL) → Phase 4 (Thesis)"],
            ["Slide 10", "Conclusion & Q&A", "Summary statement + open Q&A"],
        ],
        col_widths=[1.5*cm, 4.0*cm, w-5.5*cm]
    )

    story += file_header("📊", "tables/ — 5 Reference Tables", "Disease/Phase_1_Representation/tables/")
    story += build_table(
        ["File Name", "Table Title", "Key Information"],
        [
            ["01_dataset_taxonomy_38_classes.md", "38-Class Dataset Distribution", "Crop name, scientific name, condition status, pathogen category, approximate image counts per class, healthy vs. diseased breakdown"],
            ["02_cnn_architecture_specs.md", "CNN Layer-by-Layer Specifications", "Kernel size, stride, padding, activation, output tensor shape, parameter count per layer, mathematical formulas"],
            ["03_baseline_training_metrics.md", "Epoch-Wise Training Log", "5-epoch training/validation loss & accuracy, step time, generalization gap analysis, Phase 2 roadmap for closing the gap"],
            ["04_system_requirements_matrix.md", "Hardware & Software Stack", "Minimum vs. recommended CPU/RAM/GPU specs, library versions, Docker setup, camera requirements"],
            ["05_comparative_study_table.md", "Methodology Comparison Matrix", "Manual scouting vs. Classical ML (SVM/RF) vs. CNN across 7 evaluation dimensions"],
        ],
        col_widths=[5.0*cm, 4.5*cm, w-9.5*cm]
    )

    # ════════════════════════════════════════════════════════════════════════
    # CHAPTER 07 — DATASET REFERENCE TABLE (38 Classes Full)
    # ════════════════════════════════════════════════════════════════════════
    story += ch("Full Dataset Reference — 38 Classes", 7)
    story += body(
        "The PlantVillage dataset used in this project covers <b>14 plant species</b> across "
        "<b>38 condition classes</b> (12 healthy + 26 diseased). The dataset contains "
        "<b>54,305 RGB images</b> with standardized controlled backgrounds."
    )
    story += build_table(
        ["ID", "Crop", "Condition / Disease", "Pathogen", "Approx. Train Imgs"],
        [
            ["0",  "Apple",       "Apple Scab",           "Fungal (Venturia inaequalis)",              "~504"],
            ["1",  "Apple",       "Black Rot",             "Fungal (Botryosphaeria obtusa)",            "~496"],
            ["2",  "Apple",       "Cedar Apple Rust",      "Fungal (Gymnosporangium juniperi)",         "~220"],
            ["3",  "Apple",       "✅ Healthy",            "—",                                         "~1,316"],
            ["4",  "Blueberry",   "✅ Healthy",            "—",                                         "~1,202"],
            ["5",  "Cherry",      "Powdery Mildew",        "Fungal (Podosphaera clandestina)",          "~842"],
            ["6",  "Cherry",      "✅ Healthy",            "—",                                         "~683"],
            ["7",  "Corn",        "Cercospora / Gray Spot","Fungal (Cercospora zeae-maydis)",           "~410"],
            ["8",  "Corn",        "Common Rust",           "Fungal (Puccinia sorghi)",                  "~954"],
            ["9",  "Corn",        "Northern Leaf Blight",  "Fungal (Exserohilum turcicum)",             "~788"],
            ["10", "Corn",        "✅ Healthy",            "—",                                         "~930"],
            ["11", "Grape",       "Black Rot",             "Fungal (Guignardia bidwellii)",             "~944"],
            ["12", "Grape",       "Esca / Black Measles",  "Fungal Complex (Phaeomoniella)",            "~1,107"],
            ["13", "Grape",       "Leaf Blight",           "Fungal (Pseudocercospora)",                 "~861"],
            ["14", "Grape",       "✅ Healthy",            "—",                                         "~338"],
            ["15", "Orange",      "Huanglongbing",         "Bacterial (Candidatus Liberibacter)",       "~4,406"],
            ["16", "Peach",       "Bacterial Spot",        "Bacterial (Xanthomonas arboricola)",        "~1,838"],
            ["17", "Peach",       "✅ Healthy",            "—",                                         "~288"],
            ["18", "Pepper Bell", "Bacterial Spot",        "Bacterial (Xanthomonas campestris)",        "~798"],
            ["19", "Pepper Bell", "✅ Healthy",            "—",                                         "~1,182"],
            ["20", "Potato",      "Early Blight",          "Fungal (Alternaria solani)",                "~800"],
            ["21", "Potato",      "Late Blight",           "Oomycete (Phytophthora infestans)",         "~800"],
            ["22", "Potato",      "✅ Healthy",            "—",                                         "~122"],
            ["23", "Raspberry",   "✅ Healthy",            "—",                                         "~297"],
            ["24", "Soybean",     "✅ Healthy",            "—",                                         "~4,072"],
            ["25", "Squash",      "Powdery Mildew",        "Fungal (Erysiphe cichoracearum)",           "~1,468"],
            ["26", "Strawberry",  "Leaf Scorch",           "Fungal (Diplocarpon earlianum)",            "~887"],
            ["27", "Strawberry",  "✅ Healthy",            "—",                                         "~365"],
            ["28", "Tomato",      "Bacterial Spot",        "Bacterial (Xanthomonas vesicatoria)",       "~1,702"],
            ["29", "Tomato",      "Early Blight",          "Fungal (Alternaria solani)",                "~800"],
            ["30", "Tomato",      "Late Blight",           "Oomycete (Phytophthora infestans)",         "~1,527"],
            ["31", "Tomato",      "Leaf Mold",             "Fungal (Passalora fulva)",                  "~762"],
            ["32", "Tomato",      "Septoria Leaf Spot",    "Fungal (Septoria lycopersici)",             "~1,417"],
            ["33", "Tomato",      "Spider Mite Damage",    "Pest (Tetranychus urticae)",                "~1,341"],
            ["34", "Tomato",      "Target Spot",           "Fungal (Corynespora cassiicola)",           "~1,123"],
            ["35", "Tomato",      "Yellow Leaf Curl Virus","Viral (Begomovirus)",                       "~4,286"],
            ["36", "Tomato",      "Mosaic Virus",          "Viral (Tobamovirus)",                       "~303"],
            ["37", "Tomato",      "✅ Healthy",            "—",                                         "~1,273"],
        ],
        col_widths=[0.8*cm, 2.3*cm, 3.8*cm, 4.5*cm, w-11.4*cm]
    )

    story += build_table(
        ["Summary Metric", "Value"],
        [
            ["Total Images", "54,305"],
            ["Training Split (80%)", "43,456 images"],
            ["Validation Split (20%)", "10,849 images"],
            ["Total Plant Species", "14"],
            ["Total Condition Classes", "38"],
            ["Healthy Classes", "12"],
            ["Diseased Classes", "26"],
            ["Fungal Diseases", "16 classes"],
            ["Bacterial Diseases", "5 classes"],
            ["Viral Diseases", "2 classes"],
            ["Oomycete Diseases", "2 classes"],
            ["Pest / Mite Damage", "1 class"],
        ],
        col_widths=[5.0*cm, w-5.0*cm], accent=DARK_GRAY
    )

    # ════════════════════════════════════════════════════════════════════════
    # CHAPTER 08 — CNN ARCHITECTURE DEEP DIVE
    # ════════════════════════════════════════════════════════════════════════
    story += ch("CNN Architecture — Technical Deep Dive", 8)
    story += body(
        "A complete layer-level technical breakdown of the Sequential CNN designed for this project. "
        "This section is designed to directly support academic defense, viva questions, and "
        "methodology chapter writing in the final thesis report."
    )

    story += build_table(
        ["Layer", "Type", "Filters/Units", "Kernel", "Output Shape", "Parameters", "Role"],
        [
            ["Input", "Input", "—", "—", "(224, 224, 3)", "0", "Raw RGB leaf image tensor"],
            ["Conv2D_1", "Conv2D", "32 filters", "3×3, Valid, Stride 1", "(222, 222, 32)", "896", "Low-level: edges, color gradients, leaf veins"],
            ["MaxPool_1", "MaxPooling2D", "—", "2×2, Stride 2", "(111, 111, 32)", "0", "Spatial downsampling; translation invariance"],
            ["Conv2D_2", "Conv2D", "64 filters", "3×3, Valid, Stride 1", "(109, 109, 64)", "18,496", "Mid-level: lesion shapes, necrotic patches"],
            ["MaxPool_2", "MaxPooling2D", "—", "2×2, Stride 2", "(54, 54, 64)", "0", "Further downsampling; feature robustness"],
            ["Flatten", "Flatten", "—", "—", "(186,624)", "0", "Vectorize spatial features for dense layers"],
            ["Dense_1", "Fully Connected", "256 neurons", "Full", "(256)", "47,776,000", "High-level semantic disease patterns"],
            ["Dense_2", "Output FC", "38 neurons", "Full", "(38)", "9,766", "Class probability distribution (Softmax)"],
        ],
        col_widths=[1.8*cm, 2.5*cm, 2.0*cm, 2.5*cm, 2.5*cm, 2.3*cm, w-13.6*cm]
    )

    story += sec("Mathematical Formulas — For Thesis Writing")
    story += build_table(
        ["Formula Name", "Mathematical Expression", "Where Used"],
        [
            ["2D Convolution", "S(i,j) = Sum_m Sum_n I(i-m,j-n) * K(m,n) + b", "Conv2D layers: edge and feature extraction"],
            ["ReLU Activation", "f(x) = max(0, x)", "After each Conv2D and Dense_1 layer"],
            ["Max Pooling", "P(i,j) = max over 2x2 window of X(2i+m, 2j+n)", "After each Conv2D block: spatial reduction"],
            ["Softmax Output", "P(y=c|x) = exp(z_c) / Sum_j exp(z_j)  for c in {0..37}", "Final output layer: 38-class probabilities"],
            ["Cross-Entropy Loss", "L_CCE = -Sum_c y_c * log(y_hat_c)", "Training objective: minimized by Adam optimizer"],
            ["Adam Weight Update", "theta = theta - lr * m_hat / (sqrt(v_hat) + epsilon)", "Parameter updates during backpropagation"],
        ],
        col_widths=[3.5*cm, 6.5*cm, w-10.0*cm], accent=DARK_GRAY
    )

    story += sec("Parameter Calculation Breakdown")
    story += code_block("""\
Conv2D_1:  (kernel_h × kernel_w × in_channels + 1 bias) × filters
           = (3 × 3 × 3 + 1) × 32 = 28 × 32 = 896 parameters

Conv2D_2:  (3 × 3 × 32 + 1) × 64 = 289 × 64 = 18,496 parameters

Flatten:   54 × 54 × 64 = 186,624 feature dimensions

Dense_1:   (186,624 inputs + 1 bias) × 256 neurons = 47,776,000 parameters

Dense_2:   (256 inputs + 1 bias) × 38 neurons = 9,766 parameters

TOTAL:     896 + 18,496 + 47,776,000 + 9,766 = 47,805,158 parameters (~182 MB)
""")

    # ════════════════════════════════════════════════════════════════════════
    # CHAPTER 09 — TRAINING RESULTS
    # ════════════════════════════════════════════════════════════════════════
    story += ch("Training Results & Performance Analysis", 9)

    story += build_table(
        ["Epoch", "Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy", "Step Time", "Status"],
        [
            ["1 / 5", "0.9791", "73.28%", "0.4846", "84.65%", "108s (76ms/step)", "Rapid initial feature learning"],
            ["2 / 5", "0.2812", "91.10%", "0.4477", "86.55%", "104s (77ms/step)", "Major loss reduction (-71%)"],
            ["3 / 5", "0.1362", "95.53%", "0.4321", "88.63%", "106s (78ms/step)", "BEST val loss — peak generalization"],
            ["4 / 5", "0.0891", "97.08%", "0.5433", "87.15%", "103s (76ms/step)", "Overfitting begins (gap: +9.93%)"],
            ["5 / 5", "0.0761", "97.60%", "0.5091", "88.28%", "109s (81ms/step)", "Final baseline — gap: 9.32%"],
        ],
        col_widths=[1.4*cm, 1.8*cm, 2.8*cm, 1.8*cm, 2.8*cm, 3.5*cm, w-14.1*cm]
    )

    story += sec("Key Findings & Phase 2 Implications")
    story += body("<b>1. Convergence Speed</b>: The model reached >95% training accuracy within just 3 epochs, demonstrating strong discriminative capacity of the CNN filters over the 38-class leaf feature space.")
    story += body("<b>2. Generalization Gap</b>: By Epoch 5, a ~9.3% gap between training (97.60%) and validation (88.28%) accuracy indicates moderate overfitting — the model has memorized some training set nuances not present in validation images.")
    story += body("<b>3. Phase 2 Remediation Plan</b>:")
    items = [
        "Dropout (p=0.3–0.5) before Dense_1 — prevents neuron co-adaptation",
        "Real-time Data Augmentation (RandomFlip, RandomRotation, RandomZoom) — expands effective dataset diversity",
        "Batch Normalization after Conv layers — stabilizes and accelerates training",
        "EarlyStopping callback (monitor='val_loss', patience=5) — stops training before overfitting worsens",
        "ReduceLROnPlateau (factor=0.5, patience=3) — adapts learning rate dynamically",
    ]
    for item in items:
        story += body(f"  • {item}")

    # ════════════════════════════════════════════════════════════════════════
    # CHAPTER 10 — PHASE ROADMAP
    # ════════════════════════════════════════════════════════════════════════
    story += ch("Phase Roadmap — What Comes Next", 10)

    story += build_table(
        ["Phase", "Stage Title", "Target Duration", "Primary Deliverables", "Target Accuracy"],
        [
            ["Phase 1", "Baseline Architecture & First Representation",
             "Oct 2026",
             "Dataset taxonomy (38 classes), CNN architecture, 5-epoch baseline training, Streamlit prototype, academic synopsis, presentation slides",
             "88.28% ✅ ACHIEVED"],
            ["Phase 2", "Regularization & Optimization",
             "Oct–Nov 2026",
             "Dropout, BatchNorm, Data Augmentation, EarlyStopping, ReduceLROnPlateau, 20-epoch training, Confusion Matrix, Precision/Recall/F1 tables",
             ">94% Val Accuracy"],
            ["Phase 3", "Transfer Learning & UI Enhancement",
             "Nov–Dec 2026",
             "MobileNetV2 and ResNet50 fine-tuning, accuracy benchmark table, Streamlit UI with confidence scores and top-3 predictions, disease treatment recommendations, TFLite mobile export",
             ">96% Val Accuracy"],
            ["Phase 4", "Field Validation & Thesis Defense",
             "Dec 2026–Jan 2027",
             "Smartphone field image testing, background/lighting robustness evaluation, IEEE-format final thesis (Chapters 1–7), viva defense presentation package",
             "Production-Ready"],
        ],
        col_widths=[1.4*cm, 4.0*cm, 2.2*cm, 6.5*cm, w-14.1*cm]
    )

    story += callout(
        "🚀  To start Phase 2, simply say: \"Let's start Phase 2\" or \"Proceed to Phase 2\"\n"
        "All Phase 1 checkpoints are saved and we will pick up exactly from where we left off."
    )

    story += sec("Phase 2 Planned Code Changes (Preview)")
    story += code_block("""\
# Phase 2 — Enhanced Model Architecture (Preview)
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    layers.BatchNormalization(),          # NEW: Normalize activations
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),          # NEW
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),  # NEW: 3rd conv block
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),  # Expanded dense layer
    layers.Dropout(0.4),                   # NEW: Dropout for regularization
    layers.Dense(38, activation='softmax')
])

# Phase 2 — Data Augmentation (Preview)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.1),
])
""")

    # FINAL PAGE
    story.append(PageBreak())
    story.append(Spacer(1, 3 * cm))
    story.append(HRFlowable(width="100%", thickness=2, color=ACCENT_GREEN, spaceAfter=20))
    story.append(Paragraph("End of Phase 1 — Complete Project Guide", ParagraphStyle(
        "end", fontSize=18, alignment=TA_CENTER, textColor=DARK_BG, fontName="Helvetica-Bold", leading=28)))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        "This document covers every file and folder of the Plant Disease Detection CNN project. "
        "All Phase 1 content is complete and ready for academic review and first representation.",
        ParagraphStyle("end_body", fontSize=10, alignment=TA_CENTER, textColor=HexColor("#5D6D7E"),
                       fontName="Helvetica", leading=16)
    ))
    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width="60%", thickness=1, color=MID_GRAY, hAlign="CENTER", spaceAfter=12))
    story.append(Paragraph(
        "📌  Phase 2 begins on your command — just say: <b>'Let's start Phase 2'</b>",
        ParagraphStyle("cta", fontSize=11, alignment=TA_CENTER, textColor=ACCENT_GREEN,
                       fontName="Helvetica-Bold", leading=18)
    ))

    return story


# ── Build the PDF ──────────────────────────────────────────────────────────────
def main():
    doc = ProjectDocTemplate(
        OUTPUT_PATH,
        pagesize=A4,
        title="Plant Disease Detection CNN — Complete Project Guide",
        author="Final Year Capstone Project",
        subject="File-by-File Reference Guide — Phase 1",
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=2.0 * cm,
        bottomMargin=2.0 * cm,
    )
    story = build_document()
    doc.build(story)
    print(f"\n✅  PDF successfully generated!")
    print(f"📄  Output: {OUTPUT_PATH}")
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"📦  File Size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
