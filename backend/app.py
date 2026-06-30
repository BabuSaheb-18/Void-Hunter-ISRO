"""
==============================================================================
VHNet v2 (Multi-Task)
Flask Backend Application & PDF Generator

ISRO Hackathon 2026 - Problem Statement 10
Team : Void Hunter
Lead : Babu Saheb

Capabilities:
- Routes web traffic
- Handles telemetry uploads securely
- Triggers deep learning inference
- Generates dynamic, downloadable PDF Analysis Reports
==============================================================================
"""

import os
from pathlib import Path
from datetime import datetime
import time

# ============================================================
# FLASK
# ============================================================
from flask import Flask, render_template, request, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename

# ============================================================
# VHNET INFERENCE ENGINE
# ============================================================
from inference import enhance_infrared

# ============================================================
# PDF REPORTLAB
# ============================================================
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
)

# ============================================================
# PROJECT DIRECTORIES
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "uploads"
RESULT_FOLDER = BASE_DIR / "static" / "results"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

# ============================================================
# ALLOWED IMAGE FORMATS
# ============================================================
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}

# ============================================================
# PROJECT INFORMATION
# ============================================================
PROJECT_NAME = "VHNet v2"
TEAM_NAME = "Void Hunter"
HACKATHON = "ISRO Hackathon 2026"
FRAMEWORK = "PyTorch"
TASK_NAME = "Colorization, 2x Super-Resolution & Object Interpretation"

# ============================================================
# FLASK INITIALIZATION
# ============================================================
app = Flask(__name__)
app.secret_key = "void_hunter_isro_2026"
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024 # 25 MB Max

# ============================================================
# GLOBAL STATE (For PDF Generation)
# ============================================================
LATEST_REPORT = {}

# ============================================================
# UTILITY
# ============================================================
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ============================================================
# NAVIGATION ROUTES
# ============================================================
@app.route("/")
def overview():
    return render_template("overview.html")

@app.route("/pipeline")
def pipeline():
    return render_template("pipeline.html")

@app.route("/inference")
def inference():
    return render_template("inference.html")

# ============================================================
# IMAGE UPLOAD & INFERENCE
# ============================================================
@app.route("/upload", methods=["POST"])
def upload():
    global LATEST_REPORT

    if "image" not in request.files:
        return "No image uploaded.", 400

    file = request.files["image"]
    if file.filename == "":
        return "No file selected.", 400

    if not allowed_file(file.filename):
        return "Unsupported format. Use PNG, JPG, TIF.", 400

    # 1. Secure and save IR input
    filename = secure_filename(file.filename)
    timestamp = str(int(time.time()))
    safe_filename = f"{timestamp}_{filename}"
    
    upload_path = UPLOAD_FOLDER / safe_filename
    file.save(upload_path)

    # 2. Define Output Path
    output_filename = f"rgb_{safe_filename}"
    output_rgb_path = RESULT_FOLDER / output_filename

    # 3. Run Inference Engine
    try:
        raw_report = enhance_infrared(upload_path, output_rgb_path)
    except Exception as error:
        print(f"\n[!] Inference Error: {error}")
        return "Failed to process image through VHNet v2 backend.", 500

 # 4. Format paths for web viewing
    # Ensure a copy of the IR image exists in the static results folder so the web can see it
    import shutil
    shutil.copy(upload_path, RESULT_FOLDER / safe_filename)

    # The inference script saves absolute paths, we need relative for HTML with a leading slash
    raw_report["web_ir_path"] = f"/static/results/{safe_filename}"
    raw_report["web_rgb_path"] = f"/static/results/{output_rgb_path.name}"
    
    mask_name = f"{output_rgb_path.stem}_mask.png"
    raw_report["web_mask_path"] = f"/static/results/{mask_name}"
    
    # Save absolute paths for the PDF generator (Keep these exactly as they are)
    raw_report["abs_ir_path"] = upload_path
    raw_report["abs_rgb_path"] = output_rgb_path
    raw_report["abs_mask_path"] = RESULT_FOLDER / mask_name

    LATEST_REPORT = raw_report

    return render_template("result.html", report=LATEST_REPORT)
@app.route("/download-report")
def download_report():
    global LATEST_REPORT
    if not LATEST_REPORT:
        return "Please process an image first.", 400

    pdf_path = RESULT_FOLDER / f"VHNet_Report_{int(time.time())}.pdf"

    # Styles
    styles = getSampleStyleSheet()
    # Define custom professional styles
    title_style = styles["Title"]
    title_style.fontSize = 28
    title_style.alignment = TA_CENTER
    
    heading_style = styles["Heading2"]
    heading_style.fontSize = 16
    heading_style.alignment = TA_CENTER
    heading_style.underline = True # Header underline added
    
    normal_style = styles["BodyText"]
    normal_style.fontSize = 11

    # Tight margins to fill the page
    document = SimpleDocTemplate(str(pdf_path), pagesize=(8.5*inch, 11*inch), 
                                 topMargin=15, bottomMargin=15, leftMargin=20, rightMargin=20)
    elements = []

    # 1. Header
    elements.append(Paragraph("<u><b>VHNet v2 Analysis Report</b></u>", title_style))
    elements.append(Spacer(1, 10))

    # 2. Image Comparison (Increased to 3.25 inch to fill width)
    img_width = 3.25 * inch
    img_height = 3.25 * inch
    comparison_table = Table([
        [Paragraph("<b><u>Raw IR Input</u></b>", normal_style), Paragraph("<b><u>Enhanced HR RGB</u></b>", normal_style)],
        [Image(str(LATEST_REPORT["abs_ir_path"]), width=img_width, height=img_height),
         Image(str(LATEST_REPORT["abs_rgb_path"]), width=img_width, height=img_height)]
    ])
    # The ALIGN 'CENTER' ensures the images are centered within the cells
    comparison_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))
    elements.append(comparison_table)
    elements.append(Spacer(1, 20))

    # 3. Merged System Telemetry & Metrics
    elements.append(Paragraph("<u><b>System Telemetry & Interpretation Metrics</b></u>", heading_style))
    elements.append(Spacer(1, 10))
    
    diag_data = [
        ["Parameter", "Details"],
        ["Model_Version", LATEST_REPORT.get("model_version", "VHNet-v2")],
        ["Input_File Name", LATEST_REPORT.get("filename", "N/A")],
        ["Resolution", LATEST_REPORT.get("output_resolution", "N/A")],
        ["Inference Time", LATEST_REPORT.get("inference_time", "N/A")],
        ["PSNR", f"{LATEST_REPORT.get('psnr', 'N/A')} dB"],
        ["SSIM", str(LATEST_REPORT.get('ssim', 'N/A'))]
    ]
    diag_table = Table(diag_data, colWidths=[2.5 * inch, 3.5 * inch])
    diag_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B3D91")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    elements.append(diag_table)
    elements.append(Spacer(1, 20))

    # 4. Object Interpretation
    elements.append(Paragraph("<u><b>Geographic Object Interpretation</b></u>", heading_style))
    elements.append(Spacer(1, 10))
    
    obj_data = [["Feature Class", "Coverage (%)"]]
    for f in ["bareland", "water", "vegetation", "urban", "roads"]:
        obj_data.append([f.capitalize(), LATEST_REPORT.get(f, "0.0%")])
    
    obj_table = Table(obj_data, colWidths=[2.5 * inch, 3.5 * inch])
    obj_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#198754")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    elements.append(obj_table)

    # 5. Centered Footer
    footer_style = styles["BodyText"]
    footer_style.alignment = TA_CENTER
    elements.append(Spacer(1, 30))
    elements.append(Paragraph("<center><b>ISRO Hackathon 2026 | Team Void Hunter</b></center>", normal_style))

    document.build(elements)
    return send_file(str(pdf_path), as_attachment=True, download_name="VHNet_Analysis.pdf")

# ============================================================
# APPLICATION ENTRY POINT
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 72)
    print("                VHNet v2 Mission Control")
    print("=" * 72)
    print(f"Team             : {TEAM_NAME}")
    print(f"Event            : {HACKATHON}")
    print(f"Project          : {PROJECT_NAME}")
    print(f"Framework        : {FRAMEWORK}")
    print(f"Task             : {TASK_NAME}")
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True, use_reloader=False)