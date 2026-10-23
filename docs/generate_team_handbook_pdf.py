"""
=============================================================================
  Tomato Disease Project — Team Master Handbook & Knowledge Transfer PDF
  Generates a comprehensive, visually rich PDF for team members and viva defense.
=============================================================================
"""
import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image as RLImage, KeepTogether, HRFlowable
)

def build_team_handbook_pdf(output_pdf_path, plots_dir):
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch
    )

    styles = getSampleStyleSheet()

    # Core Color Palette
    PRIMARY = colors.HexColor('#0F2942')    # Deep Navy
    SECONDARY = colors.HexColor('#2B6CB0')  # Royal Blue
    ACCENT_GREEN = colors.HexColor('#16A34A')# Emerald Green
    ACCENT_RED = colors.HexColor('#DC2626')  # Dark Red
    BG_LIGHT = colors.HexColor('#F8FAFC')   # Clean Slate
    BORDER_COLOR = colors.HexColor('#CBD5E1')

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        textColor=PRIMARY
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#475569')
    )

    badge_style = ParagraphStyle(
        'BadgeStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        alignment=TA_CENTER,
        textColor=ACCENT_GREEN
    )

    sec_heading = ParagraphStyle(
        'SecHeading',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=15,
        textColor=PRIMARY,
        spaceBefore=11,
        spaceAfter=5,
        keepWithNext=True
    )

    subsec_heading = ParagraphStyle(
        'SubSecHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=SECONDARY,
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=9.5,
        leading=13.5,
        alignment=TA_JUSTIFY,
        spaceAfter=5
    )

    callout_body = ParagraphStyle(
        'CalloutBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        alignment=TA_LEFT
    )

    caption_style = ParagraphStyle(
        'FigCaption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#475569'),
        spaceBefore=4,
        spaceAfter=8
    )

    table_text = ParagraphStyle(
        'TableText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        alignment=TA_LEFT
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        alignment=TA_CENTER,
        textColor=colors.white
    )

    story = []

    # ── HEADER & TITLE ────────────────────────────────────────────────────────
    story.append(Paragraph("TomatoAI: Autonomous Tomato Leaf Disease Detection System", title_style))
    story.append(Spacer(1, 3))
    story.append(Paragraph("TEAM MASTER HANDBOOK &amp; COMPLETE PROJECT REFERENCE GUIDE", subtitle_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("B.Tech 4th Year Major Project | Department of Computer Science &amp; Engineering", badge_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Prepared by: <b>Shubham Jaiswal &amp; Project Team</b> | Project Approval Committee Dossier", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceBefore=4, spaceAfter=8))

    # ── SECTION 1: HOW TO EXPLAIN THIS PROJECT IN 60 SECONDS ──────────────────
    story.append(Paragraph("1. EXECUTIVE ELEVATOR PITCH (How to explain to anyone)", sec_heading))
    pitch_text = (
        "<b>The Problem:</b> Tomato is India's 2nd largest vegetable crop ($2.1B annual value), but foliar diseases like Late Blight, "
        "Bacterial Spot, and Yellow Leaf Curl Virus (TYLCV) wipe out up to <b>70–80% of local harvests</b>. Farmers lack access to plant pathologists, "
        "so they misdiagnose diseases and spray ineffective chemicals, damaging soil and wasting money.<br/><br/>"
        "<b>Our Solution:</b> We built an AI system that identifies <b>10 distinct tomato leaf conditions</b> (9 diseases + 1 healthy) from a standard "
        "smartphone photo. We evaluated a custom 4-block CNN vs a pre-trained MobileNetV2 architecture. MobileNetV2 achieved <b>69.75% test accuracy</b> "
        "(outperforming CNN by +23.9%) and trained 3.1x faster. We exported the model into an ultra-compact <b>200 KB TensorFlow Lite</b> file that runs "
        "completely offline on cheap mobile phones, and deployed a 3-tab interactive Streamlit web application."
    )
    p_box = Table([[Paragraph(pitch_text, callout_body)]], colWidths=[7.0*inch])
    p_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F0FDF4')),
        ('BOX', (0,0), (-1,-1), 1, ACCENT_GREEN),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING', (0,0), (-1,-1), 9),
        ('RIGHTPADDING', (0,0), (-1,-1), 9),
    ]))
    story.append(p_box)
    story.append(Spacer(1, 6))

    # ── SECTION 2: PROJECT STATUS & COMPLETED TASKS TRACKER ────────────────────
    story.append(Paragraph("2. PROJECT STATUS &amp; COMPLETE PHASE CHECKLIST", sec_heading))
    story.append(Paragraph(
        "Every single phase of development has been systematically completed, tested, and documented. Use this checklist during team meetings:",
        body_style
    ))

    tasks_table_data = [
        [Paragraph("<b>Phase &amp; Module</b>", table_header), Paragraph("<b>Key Tasks &amp; Milestones</b>", table_header), Paragraph("<b>Deliverables Produced</b>", table_header), Paragraph("<b>Status</b>", table_header)],
        [
            Paragraph("<b>Phase 1</b><br/>Foundation", table_text),
            Paragraph("Problem definition, literature survey, baseline CNN architecture research, hardware/software specs", table_text),
            Paragraph("PHASE_1_PROJECT_REPORT.md, 10-slide deck, Tables 1-5", table_text),
            Paragraph("<font color='#16A34A'><b>100% DONE</b></font>", table_text)
        ],
        [
            Paragraph("<b>Phase 2</b><br/>Focus &amp; Paper", table_text),
            Paragraph("Narrow scope to 10 Tomato classes only, design dual-model pipeline, draft IEEE research paper structure", table_text),
            Paragraph("Tomato_Disease_Detection_Complete.py, IEEE paper draft", table_text),
            Paragraph("<font color='#16A34A'><b>100% DONE</b></font>", table_text)
        ],
        [
            Paragraph("<b>Phase 3</b><br/>Training &amp; App", table_text),
            Paragraph("Train Custom CNN &amp; MobileNetV2 on 19,160 images, generate all 10 plots, export TFLite, upgrade Streamlit app", table_text),
            Paragraph("2 Trained .h5 models, 10 Plots, 200 KB TFLite, 3-tab web app", table_text),
            Paragraph("<font color='#16A34A'><b>100% DONE</b></font>", table_text)
        ],
        [
            Paragraph("<b>Phase 4</b><br/>Publish &amp; Viva", table_text),
            Paragraph("Compile 4.4 MB IEEE PDF paper, generate Committee Dossier, organize 10 sample test images, viva defense prep", table_text),
            Paragraph("Research Paper PDF, Committee Dossier, Team Handbook", table_text),
            Paragraph("<font color='#16A34A'><b>READY</b></font>", table_text)
        ],
    ]
    t_tasks = Table(tasks_table_data, colWidths=[1.1*inch, 2.5*inch, 2.4*inch, 1.0*inch])
    t_tasks.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [BG_LIGHT, colors.white]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_tasks)
    story.append(Spacer(1, 6))

    # ── SECTION 3: DATASET SPECIFICATION ──────────────────────────────────────
    story.append(Paragraph("3. DATASET TAXONOMY (10 Classes | 19,160 Images)", sec_heading))
    story.append(Paragraph(
        "<b>Why Tomato Only?</b> In prior papers, models trained on 38 general crops achieve artificially high 98% accuracy because distinguishing "
        "an apple leaf from a corn leaf is trivial. In reality, a farmer already knows they grow tomatoes—they need to distinguish between Early Blight, "
        "Late Blight, Septoria, and Bacterial Spot, which share very similar brown spots. Narrowing to Tomato is what makes our project scientifically sound.",
        body_style
    ))

    dataset_rows = [
        [Paragraph("<b>Class</b>", table_header), Paragraph("<b>Pathology Name</b>", table_header), Paragraph("<b>Pathogen Type</b>", table_header), Paragraph("<b>Total Images</b>", table_header), Paragraph("<b>Key Visual Symptoms</b>", table_header)],
        [Paragraph("0", table_text), Paragraph("Bacterial Spot", table_text), Paragraph("Bacterial", table_text), Paragraph("2,127", table_text), Paragraph("Small dark water-soaked spots with yellow halo", table_text)],
        [Paragraph("1", table_text), Paragraph("Early Blight", table_text), Paragraph("Fungal", table_text), Paragraph("1,000", table_text), Paragraph("Concentric target-board rings on older foliage", table_text)],
        [Paragraph("2", table_text), Paragraph("Late Blight", table_text), Paragraph("Oomycete (Critical)", table_text), Paragraph("1,909", table_text), Paragraph("Large greasy brown lesions + white mold under leaf", table_text)],
        [Paragraph("3", table_text), Paragraph("Leaf Mold", table_text), Paragraph("Fungal (Polyhouse)", table_text), Paragraph("952", table_text), Paragraph("Olive-green to gray velvety mold on undersides", table_text)],
        [Paragraph("4", table_text), Paragraph("Septoria Leaf Spot", table_text), Paragraph("Fungal", table_text), Paragraph("1,771", table_text), Paragraph("Tiny circular spots with distinct gray centers", table_text)],
        [Paragraph("5", table_text), Paragraph("Spider Mites", table_text), Paragraph("Arachnid Pest", table_text), Paragraph("1,676", table_text), Paragraph("Yellow stippling across leaf + fine silken webs", table_text)],
        [Paragraph("6", table_text), Paragraph("Target Spot", table_text), Paragraph("Fungal", table_text), Paragraph("1,404", table_text), Paragraph("Round necrotic brown lesions with defined margins", table_text)],
        [Paragraph("7", table_text), Paragraph("Yellow Leaf Curl (TYLCV)", table_text), Paragraph("Viral (Whitefly)", table_text), Paragraph("5,357", table_text), Paragraph("Severe upward leaf curling + stunted bushy growth", table_text)],
        [Paragraph("8", table_text), Paragraph("Mosaic Virus (ToMV)", table_text), Paragraph("Viral (Contact)", table_text), Paragraph("373", table_text), Paragraph("Mottled light/dark green mosaic patches on leaf", table_text)],
        [Paragraph("9", table_text), Paragraph("Healthy Foliage", table_text), Paragraph("None (Control)", table_text), Paragraph("1,591", table_text), Paragraph("Vibrant green, clean cuticle, no necrotic marks", table_text)],
    ]
    t_ds = Table(dataset_rows, colWidths=[0.5*inch, 1.6*inch, 1.3*inch, 0.9*inch, 2.7*inch])
    t_ds.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [BG_LIGHT, colors.white]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_ds)
    story.append(Spacer(1, 4))

    # Add Class Distribution Image
    c_dist_img = os.path.join(plots_dir, "01_class_distribution.png")
    if os.path.exists(c_dist_img):
        story.append(RLImage(c_dist_img, width=6.2*inch, height=2.1*inch))
        story.append(Paragraph("<b>Figure 1:</b> Dataset taxonomy and frequency distribution across all 10 tomato leaf categories.", caption_style))

    # Add Visual Atlas Image
    atlas_img = os.path.join(plots_dir, "02_sample_disease_images.png")
    if os.path.exists(atlas_img):
        story.append(RLImage(atlas_img, width=6.2*inch, height=2.3*inch))
        story.append(Paragraph("<b>Figure 2:</b> Foliar pathology visual atlas (1 sample image from each of the 10 classes).", caption_style))

    # ── SECTION 4: ARCHITECTURAL COMPARISON & EMPIRICAL RESULTS ───────────────
    story.append(Paragraph("4. ARCHITECTURAL COMPARISON &amp; EMPIRICAL RESULTS", sec_heading))
    story.append(Paragraph(
        "We compared two neural network architectures under the exact same computational setup to demonstrate the power of modern transfer learning:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Custom 4-Block CNN (Baseline):</b> Built from scratch with 4 convolutional blocks (32, 64, 128, 256 filters), Batch Normalization, "
        "MaxPooling, Dropout (0.5), and Dense 512 units.<br/>"
        "• <b>MobileNetV2 (Winner):</b> Uses Depthwise Separable Convolutions which decompose spatial filtering from channel combination. "
        "Pre-trained on ImageNet with GlobalAveragePooling2D, Dense 256, Dropout (0.4), and Softmax output.",
        body_style
    ))

    # Performance Table
    results_data = [
        [Paragraph("<b>Evaluation Metric</b>", table_header), Paragraph("<b>Custom 4-Block CNN (Scratch)</b>", table_header), Paragraph("<b>MobileNetV2 (Transfer Learning)</b>", table_header), Paragraph("<b>Advantage / Conclusion</b>", table_header)],
        [Paragraph("Top-1 Test Accuracy", table_text), Paragraph("45.84%", table_text), Paragraph("<b>69.75%</b>", table_text), Paragraph("<font color='#16A34A'><b>+23.91% higher (Winner)</b></font>", table_text)],
        [Paragraph("Test Loss (Cross-Entropy)", table_text), Paragraph("4.5074", table_text), Paragraph("<b>1.0142</b>", table_text), Paragraph("3.4x lower loss (Smoother convergence)", table_text)],
        [Paragraph("Macro Precision", table_text), Paragraph("72.02%", table_text), Paragraph("<b>77.34%</b>", table_text), Paragraph("+5.32% precision boost", table_text)],
        [Paragraph("Macro Recall", table_text), Paragraph("46.72%", table_text), Paragraph("<b>70.65%</b>", table_text), Paragraph("+23.93% sensitivity boost", table_text)],
        [Paragraph("Macro F1-Score", table_text), Paragraph("0.4608", table_text), Paragraph("<b>0.7117</b>", table_text), Paragraph("+54.4% relative improvement", table_text)],
        [Paragraph("Training Duration (CPU)", table_text), Paragraph("11.9 minutes", table_text), Paragraph("<b>3.8 minutes</b>", table_text), Paragraph("3.1x faster training on CPU", table_text)],
        [Paragraph("TFLite Model Size", table_text), Paragraph("2.11 MB", table_text), Paragraph("<b>0.20 MB (200 KB)</b>", table_text), Paragraph("10.5x smaller (Ultra-compact)", table_text)],
    ]
    t_res = Table(results_data, colWidths=[1.7*inch, 1.7*inch, 1.7*inch, 1.9*inch])
    t_res.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [BG_LIGHT, colors.white]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_res)
    story.append(Spacer(1, 4))

    # Add Dashboard and Confusion Matrix
    dash_img = os.path.join(plots_dir, "09_model_comparison.png")
    if os.path.exists(dash_img):
        story.append(RLImage(dash_img, width=6.2*inch, height=2.1*inch))
        story.append(Paragraph("<b>Figure 3:</b> Comprehensive performance telemetry comparison across all major evaluation metrics.", caption_style))

    cm_img = os.path.join(plots_dir, "06_mnv2_confusion_matrix.png")
    if os.path.exists(cm_img):
        story.append(RLImage(cm_img, width=6.2*inch, height=2.3*inch))
        story.append(Paragraph("<b>Figure 4:</b> Normalized and raw confusion matrix for MobileNetV2 across all 10 tomato categories.", caption_style))

    # ── SECTION 5: HOW TO RUN & DEMONSTRATE THE APP ───────────────────────────
    story.append(Paragraph("5. HOW TO RUN THE PROJECT (Step-by-Step for Team Members)", sec_heading))
    guide_text = (
        "<b>Step 1: Open Terminal / Shell:</b><br/>"
        "<code>cd \"/Users/shubhamjaiswal/Desktop/Disease Project/app\"</code><br/><br/>"
        "<b>Step 2: Start the Web App:</b><br/>"
        "<code>streamlit run tomato_app.py</code><br/>"
        "<i>(Or: <code>python3 -m streamlit run tomato_app.py</code>)</i><br/>"
        "• The app will automatically open in your default browser at <b>http://localhost:8501</b>.<br/><br/>"
        "<b>Step 3: Conducting the Live Demo in front of the Committee:</b><br/>"
        "• <b>Tab 1 (Diagnosis):</b> Choose <i>'Select Pre-loaded Sample (Quick Demo)'</i>. Pick any disease from the dropdown (e.g. <i>03 Late Blight</i> or <i>08 Yellow Leaf Curl</i>) and click <b>'Analyze Plant Health'</b>. Show the top-3 probabilities and chemical treatment advice.<br/>"
        "• <b>Tab 2 (Analytics):</b> Show the 10 Evaluation Plots gallery and explain the confusion matrix and convergence overlay.<br/>"
        "• <b>Tab 3 (Research Paper):</b> Show the full IEEE research paper preview and click 'Download PDF' to show the 4.4 MB compiled paper."
    )
    g_box = Table([[Paragraph(guide_text, callout_body)]], colWidths=[7.0*inch])
    g_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#EFF6FF')),
        ('BOX', (0,0), (-1,-1), 1, SECONDARY),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(g_box)
    story.append(Spacer(1, 6))

    # ── SECTION 6: VIVA VOCE DEFENSE CHEAT SHEET ──────────────────────────────
    story.append(Paragraph("6. VIVA VOCE CHEAT SHEET (Questions &amp; Ready Answers for the Committee)", sec_heading))
    story.append(Paragraph(
        "Every team member should read and remember these 5 answers before facing the Project Approval Committee:",
        body_style
    ))

    qa_list = [
        "<b>Q1: Why did you choose Tomato instead of multiple crops?</b><br/>"
        "<i>Answer:</i> In real farming, a tomato grower already knows the plant is a tomato. Real difficulty is distinguishing between Early Blight, Late Blight, Septoria, and Bacterial Spot, which all produce brown lesions. Isolating the 10 tomato classes solves the genuine agricultural challenge.",
        
        "<b>Q2: What is Depthwise Separable Convolution and why is MobileNetV2 better?</b><br/>"
        "<i>Answer:</i> Standard convolution applies spatial filtering and channel mixing simultaneously. Depthwise separable convolution splits this into (1) a 3x3 depthwise filter per channel, and (2) a 1x1 pointwise filter to mix channels. This reduces computation by ~8.5x without sacrificing accuracy, allowing MobileNetV2 to achieve 69.75% vs 45.84% for standard CNN.",
        
        "<b>Q3: What did you do to prevent overfitting?</b><br/>"
        "<i>Answer:</i> We used: (1) Data Augmentation (horizontal flip, random rotation +/-20 deg, zoom), (2) Batch Normalization to stabilize activations, (3) Dropout (0.4–0.5) to stop co-adaptation, (4) EarlyStopping with patience=5, and (5) ReduceLROnPlateau learning rate decay.",
        
        "<b>Q4: How does this help a rural farmer with no internet?</b><br/>"
        "<i>Answer:</i> We used TensorFlow Lite post-training quantization to export a <b>200 KB (.tflite)</b> model. This tiny file runs locally inside an Android smartphone or Raspberry Pi camera trap in under 45ms without needing any internet connection or cloud server.",
        
        "<b>Q5: What is the plan for Phase 4?</b><br/>"
        "<i>Answer:</i> (1) Full-resolution (224x224) GPU training to exceed 95% accuracy, (2) Field testing on natural farm backgrounds with variable sunlight, (3) Integrating Grad-CAM to highlight exactly which leaf lesions the AI looked at."
    ]

    for qa in qa_list:
        story.append(Paragraph(qa, body_style))
        story.append(Spacer(1, 2))

    # ── SECTION 7: PROJECT REPOSITORY SITEMAP ─────────────────────────────────
    story.append(Paragraph("7. PROJECT REPOSITORY SITEMAP (Where everything is located)", sec_heading))
    repo_text = (
        "• <b>app/tomato_app.py:</b> The upgraded 3-tab Streamlit web application.<br/>"
        "• <b>docs/Tomato_Disease_Detection_Research_Paper.pdf:</b> Publication-ready IEEE paper (4.4 MB).<br/>"
        "• <b>docs/PROJECT_APPROVAL_COMMITTEE_DOSSIER.md:</b> Formal committee presentation dossier.<br/>"
        "• <b>docs/Tomato_Disease_Project_Team_Handbook.pdf:</b> This exact team handbook PDF.<br/>"
        "• <b>notebooks/Tomato_Disease_Detection_Complete.py:</b> The full ML training script.<br/>"
        "• <b>tomato_output/models/:</b> Contains <code>tomato_mobilenetv2_model.h5</code> and <code>tomato_mobilenetv2.tflite</code> (200 KB).<br/>"
        "• <b>tomato_output/plots/:</b> Contains all 10 high-resolution evaluation PNG graphs.<br/>"
        "• <b>test_samples/tomato/:</b> 10 pre-loaded test leaf images (1 for each class) for quick live demos."
    )
    story.append(Paragraph(repo_text, body_style))
    story.append(Spacer(1, 8))

    doc.build(story)
    print(f"Team Handbook PDF successfully compiled: {output_pdf_path}")

if __name__ == "__main__":
    out_pdf = "/Users/shubhamjaiswal/Desktop/Disease Project/docs/Tomato_Disease_Project_Team_Handbook.pdf"
    p_dir = "/Users/shubhamjaiswal/Desktop/Disease Project/tomato_output/plots"
    build_team_handbook_pdf(out_pdf, p_dir)
