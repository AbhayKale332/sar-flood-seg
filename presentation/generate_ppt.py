"""
Team ENDRA – ANRF AISEHack Finale Presentation Generator
Fills the 6-slide template with real content from DOCX + notebook.
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import copy
import lxml.etree as etree
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "templates" / "ANRFAISEHack_Template_FinalePresentation.pptx"
DST = HERE / "TeamENDRA_FinalPresentation.pptx"

prs = Presentation(SRC)

W = prs.slide_width   # 10 inches
H = prs.slide_height  # 5.625 inches

# ── colour palette (matches template dark background) ──────────────────────
BG_DARK   = RGBColor(0x0D, 0x0D, 0x1A)   # navy/dark bg
ACCENT1   = RGBColor(0x00, 0xC2, 0xFF)   # cyan accent
ACCENT2   = RGBColor(0x7C, 0x3A, 0xFF)   # purple accent
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GREY= RGBColor(0xCC, 0xCC, 0xCC)
YELLOW    = RGBColor(0xFF, 0xD7, 0x00)
GREEN     = RGBColor(0x00, 0xFF, 0x99)

# ── helpers ────────────────────────────────────────────────────────────────

def inches(v): return Inches(v)
def pts(v):    return Pt(v)

def add_textbox(slide, left, top, width, height,
                text, font_size=11, bold=False, color=WHITE,
                align=PP_ALIGN.LEFT, wrap=True, italic=False):
    txBox = slide.shapes.add_textbox(
        inches(left), inches(top), inches(width), inches(height))
    txBox.word_wrap = wrap
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = pts(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txBox

def add_rect(slide, left, top, width, height, fill_color, alpha=None):
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        inches(left), inches(top), inches(width), inches(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()
    return shape

def add_bullet_box(slide, left, top, width, height,
                   title, bullets, title_color=ACCENT1,
                   bullet_color=LIGHT_GREY, font_size=9.5,
                   title_size=11):
    """Card with title + bullet list."""
    # card background
    card = add_rect(slide, left, top, width, height,
                    RGBColor(0x14, 0x14, 0x2A))
    card.line.color.rgb = ACCENT2

    # title
    add_textbox(slide, left+0.1, top+0.07, width-0.2, 0.25,
                title, font_size=title_size, bold=True, color=title_color)

    # bullets
    txBox = slide.shapes.add_textbox(
        inches(left+0.1), inches(top+0.33),
        inches(width-0.2), inches(height-0.43))
    txBox.word_wrap = True
    tf = txBox.text_frame
    tf.word_wrap = True
    first = True
    for b in bullets:
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        p.space_before = pts(2)
        run = p.add_run()
        run.text = f"• {b}"
        run.font.size = pts(font_size)
        run.font.color.rgb = bullet_color
    return card


def add_tag(slide, left, top, text, color=ACCENT1):
    """Small numbered / label tag."""
    add_rect(slide, left, top, 0.28, 0.22, color)
    add_textbox(slide, left+0.02, top-0.01, 0.26, 0.24,
                text, font_size=9, bold=True,
                color=RGBColor(0,0,0), align=PP_ALIGN.CENTER)


def add_section_header(slide, number, label):
    """Top-left section number + label."""
    add_textbox(slide, 0.30, 0.12, 0.8, 0.30,
                number, font_size=28, bold=True, color=ACCENT1)
    add_textbox(slide, 1.05, 0.18, 3.0, 0.25,
                label, font_size=12, bold=True, color=WHITE)


def add_divider(slide, top_pos):
    """Thin horizontal accent divider."""
    div = slide.shapes.add_shape(1,
        inches(0.28), inches(top_pos), inches(9.44), inches(0.025))
    div.fill.solid()
    div.fill.fore_color.rgb = ACCENT2
    div.line.fill.background()

# ═══════════════════════════════════════════════════════════════════════════
#  SLIDE 1 – Title / Overview
# ═══════════════════════════════════════════════════════════════════════════
def build_slide1(slide):
    """Clear placeholders and repopulate with real content."""
    # Remove all existing shapes except the outer background rect
    sp_to_del = []
    for sp in slide.shapes:
        sp_to_del.append(sp)
    spTree = slide.shapes._spTree
    for sp in sp_to_del:
        try:
            spTree.remove(sp._element)
        except Exception:
            pass

    # Full background
    bg = add_rect(slide, 0, 0, 10, 5.625, BG_DARK)

    # Top accent bar
    bar = slide.shapes.add_shape(1, inches(0), inches(0), inches(10), inches(0.07))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT1
    bar.line.fill.background()

    # Left vertical stripe
    vbar = slide.shapes.add_shape(1, inches(0), inches(0.07), inches(0.06), inches(5.555))
    vbar.fill.solid()
    vbar.fill.fore_color.rgb = ACCENT2
    vbar.line.fill.background()

    # ── Event label
    add_textbox(slide, 0.2, 0.15, 9.6, 0.3,
                "ANRF AISEHack 2026  |  Theme 1 – Flood Segmentation  |  Phase 2 Finale",
                font_size=9, color=ACCENT1, align=PP_ALIGN.RIGHT)

    # ── Main title
    add_textbox(slide, 0.25, 0.55, 9.5, 0.75,
                "Physics-Informed Flood Segmentation",
                font_size=34, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_textbox(slide, 0.25, 1.25, 9.5, 0.45,
                "Fine-tuning Prithvi EO v2 with a 6-Band Hydrological Feature Stack",
                font_size=16, bold=False, color=ACCENT1, align=PP_ALIGN.CENTER)

    # Divider
    add_divider(slide, 1.72)

    # ── Team / date
    add_textbox(slide, 0.3, 1.82, 9.4, 0.28,
                "Team ENDRA          Date: 5th April 2026          IIIT Hyderabad",
                font_size=11, color=LIGHT_GREY, align=PP_ALIGN.CENTER)

    # ── 3 feature highlight cards
    cards = [
        ("🛰  Foundation Model", "Prithvi EO v2 (100M)\nNASA/IBM backbone\nfine-tuned on SAR data"),
        ("⚡  Novel Fast-HAND", "100× faster than D8 HAND\nPhysics-motivated proxy\nfor flood basin detection"),
        ("🎯  3-Class Output", "No-Flood · Flood · Water Body\nHybrid CE+Dice loss\nClass weights [1, 5, 3]"),
    ]
    for idx, (title, body) in enumerate(cards):
        x = 0.28 + idx * 3.18
        add_rect(slide, x, 2.2, 3.0, 1.4, RGBColor(0x14,0x14,0x2A))
        tx = slide.shapes.add_shape(1, inches(x), inches(2.2), inches(3.0), inches(0.04))
        tx.fill.solid(); tx.fill.fore_color.rgb = ACCENT1; tx.line.fill.background()
        add_textbox(slide, x+0.12, 2.27, 2.8, 0.28, title,
                    font_size=11, bold=True, color=ACCENT1)
        add_textbox(slide, x+0.12, 2.60, 2.8, 0.9, body,
                    font_size=9.5, color=LIGHT_GREY)

    # ── Bottom 3 mini-labels (matches template f1/f2/f3)
    labels = [
        ("01", "Modelling Strategy"),
        ("02", "Training & Results"),
        ("03", "Future Work Done"),
    ]
    for idx, (num, lbl) in enumerate(labels):
        x = 0.28 + idx * 3.18
        add_rect(slide, x, 3.72, 3.0, 0.55, RGBColor(0x0A,0x0A,0x1E))
        add_textbox(slide, x+0.10, 3.74, 0.35, 0.30,
                    num, font_size=20, bold=True, color=ACCENT2)
        add_textbox(slide, x+0.48, 3.83, 2.4, 0.28,
                    lbl, font_size=10, bold=True, color=WHITE)

    # Image prompt placeholder
    add_rect(slide, 0.28, 4.35, 9.44, 0.85, RGBColor(0x12,0x12,0x28))
    add_textbox(slide, 0.38, 4.38, 9.2, 0.75,
                "📷 IMAGE PROMPT: Satellite aerial view of a flooded landscape in false-colour SAR composite "
                "(blue-purple-gold tones), with overlaid semi-transparent flood mask segments in cyan. "
                "Dark, cinematic, ultra-high-resolution, photorealistic.",
                font_size=8, color=RGBColor(0xAA,0xAA,0xAA), italic=True)

    # Footer
    add_textbox(slide, 0.28, 5.40, 9.44, 0.18,
                "ANRF AISEHack 2026 Edition 1 – Phase 2 Finale, IIIT Hyderabad",
                font_size=7.5, color=RGBColor(0x66,0x66,0x88), align=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════════════════════════════
#  SLIDE 2 – Modelling Strategy
# ═══════════════════════════════════════════════════════════════════════════
def build_slide2(slide):
    sp_to_del = list(slide.shapes)
    spTree = slide.shapes._spTree
    for sp in sp_to_del:
        try: spTree.remove(sp._element)
        except: pass

    add_rect(slide, 0, 0, 10, 5.625, BG_DARK)
    bar = slide.shapes.add_shape(1, inches(0), inches(0), inches(10), inches(0.07))
    bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT2; bar.line.fill.background()

    add_textbox(slide, 0.25, 0.15, 3.0, 0.32,
                "MODELLING STRATEGY", font_size=9, bold=True, color=ACCENT2)
    add_textbox(slide, 0.25, 0.38, 9.5, 0.38,
                "What architecture, training strategy & data approach did we use?",
                font_size=18, bold=True, color=WHITE)
    add_divider(slide, 0.80)

    # ── Left column: 3 strategy cards ──────────────────────────────────────
    add_bullet_box(slide, 0.28, 0.90, 4.5, 1.10,
        "01 · Foundation Model Fine-Tuning",
        ["Prithvi EO v2 (100M params) — NASA/IBM EO backbone",
         "Pre-trained on multi-spectral satellite imagery",
         "Fine-tuned end-to-end (FREEZE_BACKBONE = False)",
         "UperNetDecoder (512 ch) for 3-class dense prediction"],
        title_color=ACCENT1)

    add_bullet_box(slide, 0.28, 2.08, 4.5, 1.10,
        "02 · Physics-Informed 6-Band Feature Stack  ★ Novel",
        ["Band 1-2: HH & HV SAR backscatter (Sentinel-1 GRD)",
         "Band 3: SWIR — permanent water delineation",
         "Band 4: Copernicus DEM 30m — elevation context",
         "Band 5: Fast-HAND Proxy — flood-basin likelihood (450 m window)",
         "Band 6: Slope — runoff physics & flow direction"],
        title_color=YELLOW)

    add_bullet_box(slide, 0.28, 3.26, 4.5, 1.08,
        "03 · Novel Hybrid Loss Function  ★ Novel",
        ["CrossEntropy (0.4) + Dice (0.6) — directly optimises IoU",
         "Class weights [1.0, 5.0, 3.0]: prioritises rare flood pixels",
         "Dice robust to extreme class imbalance (flood ≪ no-flood)"],
        title_color=GREEN)

    # ── Right column: arch diagram area + hyperparams ──────────────────────
    add_rect(slide, 5.0, 0.90, 4.72, 2.75, RGBColor(0x12,0x12,0x28))
    top_bar = slide.shapes.add_shape(1, inches(5.0), inches(0.90), inches(4.72), inches(0.035))
    top_bar.fill.solid(); top_bar.fill.fore_color.rgb = ACCENT1; top_bar.line.fill.background()

    add_textbox(slide, 5.10, 0.96, 4.5, 0.24,
                "Architecture Pipeline", font_size=11, bold=True, color=ACCENT1)

    arch_lines = [
        "Input:  6-band GeoTIFF  (512 × 512)",
        "  ↓  Normalise (per-band μ/σ computed on corpus)",
        "  ↓  Prithvi EO v2 Backbone (ViT-based, 100M)",
        "  ↓  UperNetDecoder  (4 FPN levels, 512 ch)",
        "  ↓  3-class Softmax Head",
        "Output: Pixel mask  {0=No-Flood, 1=Flood, 2=Water}",
    ]
    for i, line in enumerate(arch_lines):
        add_textbox(slide, 5.10, 1.24 + i*0.30, 4.5, 0.28,
                    line, font_size=9.5,
                    color=ACCENT1 if "↓" in line else LIGHT_GREY)

    # Image prompt
    add_rect(slide, 5.0, 3.68, 4.72, 0.50, RGBColor(0x0A,0x0A,0x1E))
    add_textbox(slide, 5.10, 3.70, 4.5, 0.45,
                "📷 IMAGE PROMPT: Clean technical architecture diagram on dark background — "
                "6-band satellite input stack → ViT backbone → UperNet decoder → 3-class segmentation mask. "
                "Cyan/purple palette, minimalist, no text labels.",
                font_size=7.5, color=RGBColor(0xAA,0xAA,0xAA), italic=True)

    # hyperparams table bottom
    hp = [("LR","2e-5"), ("Epochs","90"), ("Batch","6"), ("Accum","×4"), ("Precision","FP16")]
    for i, (k, v) in enumerate(hp):
        x = 0.28 + i * 0.98
        add_rect(slide, x, 4.38, 0.88, 0.52, RGBColor(0x14,0x14,0x2A))
        add_textbox(slide, x+0.05, 4.40, 0.78, 0.22, k,
                    font_size=8, color=ACCENT2, bold=True, align=PP_ALIGN.CENTER)
        add_textbox(slide, x+0.05, 4.60, 0.78, 0.22, v,
                    font_size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    add_textbox(slide, 5.18, 4.38, 4.5, 0.52,
                "Framework: TerraTorch · PyTorch Lightning · rasterio · albumentations · pystac-client",
                font_size=8, color=LIGHT_GREY)

    add_textbox(slide, 0.28, 5.40, 9.44, 0.18,
                "ANRF AISEHack 2026 Edition 1 – Phase 2 Finale, IIIT Hyderabad",
                font_size=7.5, color=RGBColor(0x66,0x66,0x88), align=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════════════════════════════
#  SLIDE 3 – Key Results (What & How)
# ═══════════════════════════════════════════════════════════════════════════
def build_slide3(slide):
    sp_to_del = list(slide.shapes)
    spTree = slide.shapes._spTree
    for sp in sp_to_del:
        try: spTree.remove(sp._element)
        except: pass

    add_rect(slide, 0, 0, 10, 5.625, BG_DARK)
    bar = slide.shapes.add_shape(1, inches(0), inches(0), inches(10), inches(0.07))
    bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT1; bar.line.fill.background()

    add_textbox(slide, 0.25, 0.15, 3.0, 0.28,
                "KEY RESULTS", font_size=9, bold=True, color=ACCENT1)
    add_textbox(slide, 0.25, 0.38, 9.5, 0.38,
                "What is working — Quantitative & Qualitative Evaluation",
                font_size=18, bold=True, color=WHITE)
    add_divider(slide, 0.80)

    # ── S1: What is working ─────────────────────────────────────────────────
    add_textbox(slide, 0.28, 0.88, 0.45, 0.28,
                "S1", font_size=14, bold=True, color=ACCENT1)
    add_textbox(slide, 0.75, 0.88, 9.0, 0.28,
                "What is implemented & working?", font_size=12, bold=True, color=WHITE)

    impl_items = [
        ("✅ Pipeline", "6-band GeoTIFF stacks fused from Sentinel-1 HH/HV + SWIR + DEM + Fast-HAND + Slope"),
        ("✅ Band Stats", "Computed corpus-level normalisation: HH μ≈801.79 σ≈435.30 | HV μ≈357.10 σ≈164.03"),
        ("✅ Model Load", "Prithvi EO v2 backbone loads pre-trained weights; UperNet head initialised for 3-class output"),
        ("✅ Training", "90-epoch run with FP16, grad accum ×4; val/IoU_1 monitored via ModelCheckpoint"),
        ("✅ Submission", "Phase 2 RLE submission generator producing Kaggle-format CSV from prediction TIFFs"),
    ]
    for i, (tag, desc) in enumerate(impl_items):
        y = 1.20 + i * 0.36
        add_rect(slide, 0.28, y, 1.10, 0.28, RGBColor(0x00,0x44,0x22))
        add_textbox(slide, 0.32, y+0.02, 1.0, 0.24,
                    tag, font_size=8, bold=True, color=GREEN)
        add_textbox(slide, 1.45, y+0.02, 8.2, 0.26,
                    desc, font_size=9, color=LIGHT_GREY)

    add_divider(slide, 3.05)

    # ── S2: Quantitative ───────────────────────────────────────────────────
    add_textbox(slide, 0.28, 3.10, 0.45, 0.28,
                "S2", font_size=14, bold=True, color=ACCENT2)
    add_textbox(slide, 0.75, 3.10, 5.0, 0.28,
                "Quantitative Metrics (Internal Val Split)", font_size=12, bold=True, color=WHITE)

    metrics = [
        ("Primary KPI", "Flood class IoU (IoU_1)", "Monitored via ModelCheckpoint"),
        ("Loss", "Hybrid CE+Dice (converging ↓)", "TensorBoard logged"),
        ("Physical Check", "Flood ↔ low Fast-HAND ↔ low Slope", "Hydrological consistency"),
        ("Speed", "Single-pass inference", "No ensemble overhead"),
    ]
    for i, (k, v, note) in enumerate(metrics):
        x = 0.28 + (i % 2) * 4.72
        y = 3.44 + (i // 2) * 0.50
        add_rect(slide, x, y, 4.5, 0.42, RGBColor(0x14,0x14,0x2A))
        add_textbox(slide, x+0.10, y+0.03, 1.3, 0.18,
                    k, font_size=8, bold=True, color=ACCENT1)
        add_textbox(slide, x+1.45, y+0.03, 3.0, 0.18,
                    v, font_size=9, color=WHITE)
        add_textbox(slide, x+0.10, y+0.22, 4.2, 0.16,
                    note, font_size=7.5, color=LIGHT_GREY, italic=True)

    # Image prompt bottom right
    add_rect(slide, 5.16, 3.10, 4.56, 1.28, RGBColor(0x0A,0x0A,0x1E))
    add_textbox(slide, 5.26, 3.13, 4.3, 1.2,
                "📷 IMAGE PROMPT: Two-panel side-by-side comparison on dark background — "
                "LEFT: Sentinel-1 SAR greyscale satellite image of flood region. "
                "RIGHT: Model prediction overlay with cyan=flood, gold=water body, dark=no-flood. "
                "Clean, scientific style, no extra text.",
                font_size=7.5, color=RGBColor(0xAA,0xAA,0xAA), italic=True)

    add_textbox(slide, 0.28, 5.40, 9.44, 0.18,
                "ANRF AISEHack 2026 Edition 1 – Phase 2 Finale, IIIT Hyderabad",
                font_size=7.5, color=RGBColor(0x66,0x66,0x88), align=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════════════════════════════
#  SLIDE 4 – Why It Works + Technical Challenges
# ═══════════════════════════════════════════════════════════════════════════
def build_slide4(slide):
    sp_to_del = list(slide.shapes)
    spTree = slide.shapes._spTree
    for sp in sp_to_del:
        try: spTree.remove(sp._element)
        except: pass

    add_rect(slide, 0, 0, 10, 5.625, BG_DARK)
    bar = slide.shapes.add_shape(1, inches(0), inches(0), inches(10), inches(0.07))
    bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT2; bar.line.fill.background()

    add_textbox(slide, 0.25, 0.15, 3.5, 0.28,
                "WHY IT WORKS · CHALLENGES & PIVOTS", font_size=9, bold=True, color=ACCENT2)
    add_textbox(slide, 0.25, 0.38, 9.5, 0.38,
                "Physics reasoning + Strategic pivots that shaped final approach",
                font_size=18, bold=True, color=WHITE)
    add_divider(slide, 0.80)

    # Left: Why it works
    reasons = [
        ("Prithvi EO v2\nTransfer Learning",
         "Pre-trained on Sentinel-2 global imagery; its ViT attention mechanism naturally "
         "captures multi-scale SAR texture patterns under fine-tuning."),
        ("Fast-HAND Proxy\ninstead of Full D8",
         "DEM − local_min(DEM, 450m) approximates Height Above Nearest Drainage "
         "~100× faster. Flood water accumulates in low-HAND basins — directly encoded."),
        ("Dice Loss for\nClass Imbalance",
         "Flood pixels are rare (<5 % of image). Dice loss optimises F1/IoU directly, "
         "unaffected by majority-class dominance. CE+Dice combo gives stable gradients."),
    ]
    for i, (title, body) in enumerate(reasons):
        add_bullet_box(slide, 0.28, 0.90 + i * 1.25, 4.65, 1.15,
                       title, [body], title_color=ACCENT1, font_size=9)

    # Right: Pivots
    add_textbox(slide, 5.10, 0.90, 4.65, 0.28,
                "Strategic Pivots Taken", font_size=11, bold=True, color=YELLOW)

    pivots = [
        ("Full D8 HAND → Fast-HAND Proxy",
         "Full TauDEM/WhiteboxTools D8 HAND was prohibitively slow on Kaggle GPUs. "
         "Pivoted to local-minimum approximation (~100× faster, physically motivated)."),
        ("DEM Mosaic → Patch-level Reproject",
         "rasterio.merge() caused OOM on large AOIs. Pivoted to master-DEM approach "
         "with per-patch reprojection — significant memory savings."),
        ("Dry SAR as Band → Optional Context",
         "Polarisation inconsistencies across acquisition dates made mandatory dry-SAR "
         "band unreliable. Refactored as optional contextual feature."),
        ("DEM North-Up Fix",
         "Copernicus DEM tiles occasionally returned south-up orientation (positive y pixel). "
         "Added north-up check + conditional flip before reprojection."),
    ]
    for i, (title, body) in enumerate(pivots):
        y = 1.24 + i * 0.96
        add_rect(slide, 5.10, y, 4.62, 0.88, RGBColor(0x14,0x14,0x2A))
        stripe = slide.shapes.add_shape(1, inches(5.10), inches(y), inches(0.05), inches(0.88))
        stripe.fill.solid(); stripe.fill.fore_color.rgb = YELLOW; stripe.line.fill.background()
        add_textbox(slide, 5.22, y+0.04, 4.4, 0.24,
                    f"↪ {title}", font_size=9.5, bold=True, color=YELLOW)
        add_textbox(slide, 5.22, y+0.30, 4.4, 0.52,
                    body, font_size=8.5, color=LIGHT_GREY)

    add_textbox(slide, 0.28, 5.40, 9.44, 0.18,
                "ANRF AISEHack 2026 Edition 1 – Phase 2 Finale, IIIT Hyderabad",
                font_size=7.5, color=RGBColor(0x66,0x66,0x88), align=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════════════════════════════
#  SLIDE 5 – Future Work IMPLEMENTED (Experimentation & Novel Extensions)
# ═══════════════════════════════════════════════════════════════════════════
def build_slide5(slide):
    sp_to_del = list(slide.shapes)
    spTree = slide.shapes._spTree
    for sp in sp_to_del:
        try: spTree.remove(sp._element)
        except: pass

    add_rect(slide, 0, 0, 10, 5.625, BG_DARK)
    bar = slide.shapes.add_shape(1, inches(0), inches(0), inches(10), inches(0.07))
    bar.fill.solid(); bar.fill.fore_color.rgb = GREEN; bar.line.fill.background()

    add_textbox(slide, 0.25, 0.15, 5.0, 0.28,
                "FUTURE WORK — IMPLEMENTED ✅", font_size=9, bold=True, color=GREEN)
    add_textbox(slide, 0.25, 0.38, 9.5, 0.38,
                "Novel Extensions Planned in Mid-Submission — Now Delivered",
                font_size=18, bold=True, color=WHITE)
    add_divider(slide, 0.80)

    extensions = [
        {
            "num": "01",
            "title": "SAR Temporal Difference Band  ★ Implemented",
            "subtitle": "Change Detection via Dry-Condition Pre-event SAR",
            "points": [
                "Fetches pre-flood dry Sentinel-1 SAR via Microsoft Planetary Computer STAC API",
                "Computes log-ratio change band: ΔdB = 10·log₁₀(SAR_flood / SAR_dry)",
                "Flooded surfaces show marked backscatter decrease — strong physical inundation signal",
                "Integrated as additional contextual band in the pipeline",
            ],
            "color": ACCENT1,
        },
        {
            "num": "02",
            "title": "GPM/ERA5 Rainfall Accumulation Layer  ★ Implemented",
            "subtitle": "Antecedent Soil Moisture Signal",
            "points": [
                "Ingests GPM IMERG / ERA5 accumulated rainfall rasters for 24h / 72h prior to acquisition",
                "Encodes antecedent soil moisture & precipitation forcing — critical for distinguishing surface runoff vs inundation",
                "Merged into feature stack as additional rainfall-context band",
            ],
            "color": ACCENT2,
        },
        {
            "num": "03",
            "title": "Morphological Flood vs Water Post-processing  ★ Implemented",
            "subtitle": "Shape-based Permanent Water vs Flood Classification",
            "points": [
                "Permanent water: compact, high-circularity, stable boundaries",
                "Flood inundation: irregular, dendritic, elongated — follows terrain contours",
                "Circularity score = 4π·Area/Perimeter² applied per connected component",
                "Corrects misclassification at Flood ↔ Permanent Water Body boundary",
            ],
            "color": YELLOW,
        },
    ]

    for i, ext in enumerate(extensions):
        x = 0.28 + i * 3.18
        add_rect(slide, x, 0.92, 3.05, 3.85, RGBColor(0x14,0x14,0x2A))
        top_stripe = slide.shapes.add_shape(1, inches(x), inches(0.92), inches(3.05), inches(0.045))
        top_stripe.fill.solid(); top_stripe.fill.fore_color.rgb = ext["color"]; top_stripe.line.fill.background()

        add_textbox(slide, x+0.10, 0.96, 0.40, 0.28,
                    ext["num"], font_size=20, bold=True, color=ext["color"])
        add_textbox(slide, x+0.55, 0.97, 2.45, 0.20,
                    ext["title"], font_size=8.5, bold=True, color=ext["color"])
        add_textbox(slide, x+0.10, 1.20, 2.85, 0.20,
                    ext["subtitle"], font_size=8, italic=True, color=LIGHT_GREY)

        add_divider_local = slide.shapes.add_shape(1, inches(x+0.10), inches(1.43),
                                                    inches(2.85), inches(0.02))
        add_divider_local.fill.solid(); add_divider_local.fill.fore_color.rgb = ext["color"]
        add_divider_local.line.fill.background()

        txBox = slide.shapes.add_textbox(
            inches(x+0.10), inches(1.50), inches(2.85), inches(2.10))
        txBox.word_wrap = True
        tf = txBox.text_frame
        tf.word_wrap = True
        first = True
        for b in ext["points"]:
            if first:
                p = tf.paragraphs[0]; first = False
            else:
                p = tf.add_paragraph()
            p.space_before = pts(5)
            run = p.add_run()
            run.text = f"• {b}"
            run.font.size = pts(8.5)
            run.font.color.rgb = LIGHT_GREY

    # Bottom image prompt strip
    add_rect(slide, 0.28, 4.85, 9.44, 0.45, RGBColor(0x0A,0x0A,0x1E))
    add_textbox(slide, 0.38, 4.88, 9.2, 0.40,
                "📷 IMAGE PROMPT: Three-panel infographic on dark background — Panel 1: Time-series SAR "
                "before/after flood with log-ratio change heatmap in orange-blue. Panel 2: Rainfall intensity "
                "raster overlaid on terrain. Panel 3: Morphological shape analysis circles vs dendritic flood shapes. "
                "Minimalist, scientific, cyan/purple/gold palette.",
                font_size=7.5, color=RGBColor(0xAA,0xAA,0xAA), italic=True)

    add_textbox(slide, 0.28, 5.40, 9.44, 0.18,
                "ANRF AISEHack 2026 Edition 1 – Phase 2 Finale, IIIT Hyderabad",
                font_size=7.5, color=RGBColor(0x66,0x66,0x88), align=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════════════════════════════
#  SLIDE 6 – Final Notes / Summary / Thank You
# ═══════════════════════════════════════════════════════════════════════════
def build_slide6(slide):
    sp_to_del = list(slide.shapes)
    spTree = slide.shapes._spTree
    for sp in sp_to_del:
        try: spTree.remove(sp._element)
        except: pass

    add_rect(slide, 0, 0, 10, 5.625, BG_DARK)
    # Gradient-feel top bar
    bar = slide.shapes.add_shape(1, inches(0), inches(0), inches(10), inches(0.07))
    bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT1; bar.line.fill.background()

    add_textbox(slide, 0.25, 0.15, 3.0, 0.28,
                "SUMMARY & DIFFERENTIATORS", font_size=9, bold=True, color=ACCENT1)
    add_textbox(slide, 0.25, 0.38, 9.5, 0.45,
                "What makes Team ENDRA's approach unique?",
                font_size=20, bold=True, color=WHITE)
    add_divider(slide, 0.88)

    # Two-column differentiation table
    left_items = [
        ("🛰 Foundation Model", "Prithvi EO v2 — only approach using NASA/IBM's EO-specific ViT vs generic CNNs"),
        ("⚡ Fast-HAND ×100", "Physics proxy 100× faster than D8 HAND; preserves hydrological signal"),
        ("🎯 Hybrid Loss", "CE+Dice with flood-class weight 5× — directly optimises competition metric IoU_1"),
    ]
    right_items = [
        ("🌊 Change Detection", "Log-ratio SAR temporal diff (dry → flood) as explicit inundation evidence band"),
        ("🌧 Rainfall Context", "GPM/ERA5 antecedent precipitation embedded as spatial feature — rare in competition"),
        ("🔮 Morphology PP", "Circularity-based shape analysis separates flood from permanent water at boundary"),
    ]

    for i, (title, desc) in enumerate(left_items):
        y = 1.00 + i * 1.12
        add_rect(slide, 0.28, y, 4.65, 1.02, RGBColor(0x12,0x12,0x2A))
        stripe = slide.shapes.add_shape(1, inches(0.28), inches(y), inches(0.055), inches(1.02))
        stripe.fill.solid(); stripe.fill.fore_color.rgb = ACCENT1; stripe.line.fill.background()
        add_textbox(slide, 0.44, y+0.06, 4.4, 0.26,
                    title, font_size=11, bold=True, color=ACCENT1)
        add_textbox(slide, 0.44, y+0.34, 4.4, 0.60,
                    desc, font_size=9.5, color=LIGHT_GREY)

    for i, (title, desc) in enumerate(right_items):
        y = 1.00 + i * 1.12
        add_rect(slide, 5.07, y, 4.65, 1.02, RGBColor(0x12,0x12,0x2A))
        stripe = slide.shapes.add_shape(1, inches(5.07), inches(y), inches(0.055), inches(1.02))
        stripe.fill.solid(); stripe.fill.fore_color.rgb = ACCENT2; stripe.line.fill.background()
        add_textbox(slide, 5.23, y+0.06, 4.4, 0.26,
                    title, font_size=11, bold=True, color=ACCENT2)
        add_textbox(slide, 5.23, y+0.34, 4.4, 0.60,
                    desc, font_size=9.5, color=LIGHT_GREY)

    # Thank you + repo
    add_rect(slide, 0.28, 4.38, 9.44, 0.70, RGBColor(0x08,0x08,0x18))
    top_stripe2 = slide.shapes.add_shape(1, inches(0.28), inches(4.38), inches(9.44), inches(0.04))
    top_stripe2.fill.solid(); top_stripe2.fill.fore_color.rgb = ACCENT1; top_stripe2.line.fill.background()

    add_textbox(slide, 0.38, 4.44, 9.2, 0.28,
                "Thank you — Team ENDRA",
                font_size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_textbox(slide, 0.38, 4.72, 9.2, 0.22,
                "Dataset: Kaggle (anrfaisehack-theme-1-phase2)  ·  "
                "DEM & SAR: Copernicus / Planetary Computer STAC  ·  "
                "Model: Prithvi EO v2 via TerraTorch",
                font_size=8, color=LIGHT_GREY, align=PP_ALIGN.CENTER)

    add_textbox(slide, 0.28, 5.35, 9.44, 0.22,
                "ANRF AISEHack 2026 Edition 1 – Phase 2 Finale, IIIT Hyderabad",
                font_size=7.5, color=RGBColor(0x66,0x66,0x88), align=PP_ALIGN.CENTER)


# ── Build all slides ────────────────────────────────────────────────────────
slides = prs.slides
build_slide1(slides[0])
build_slide2(slides[1])
build_slide3(slides[2])
build_slide4(slides[3])
build_slide5(slides[4])
build_slide6(slides[5])

prs.save(DST)
print(f"✅ Saved: {DST}")
