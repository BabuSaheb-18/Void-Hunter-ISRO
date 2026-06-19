from flask import (
    Flask,
    render_template,
    request,
    send_file
)

from werkzeug.utils import secure_filename

import os
from datetime import datetime
from inference import enhance_infrared
import inference

print("Inference module:", inference.__file__)
from reportlab.platypus import KeepTogether
# ==========================================================
# PDF
# ==========================================================

from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    Image,
    PageBreak
)

from reportlab.lib import colors

from reportlab.lib.styles import getSampleStyleSheet

from reportlab.lib.units import inch


# ==========================================================
# Flask
# ==========================================================

app = Flask(__name__)
BASE_DIR = os.path.dirname(

    os.path.dirname(

        os.path.abspath(__file__)

    )

)

# ==========================================================
# Configuration
# ==========================================================

UPLOAD_FOLDER = "uploads"

RESULT_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "results"
)
ALLOWED_EXTENSIONS = {

    "png",

    "jpg",

    "jpeg",

    "bmp",

    "tif",

    "tiff"

}

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    RESULT_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ==========================================================
# Globals
# ==========================================================

LATEST_FILE = ""

LATEST_REPORT = {}


# ==========================================================
# Utility
# ==========================================================

def allowed_file(filename):

    return (

        "." in filename

        and

        filename.rsplit(".", 1)[1].lower()

        in ALLOWED_EXTENSIONS

    )# ==========================================================
# Routes
# ==========================================================

@app.route("/")
def home():

    return render_template(
        "overview.html"
    )


@app.route("/pipeline")
def pipeline():

    return render_template(
        "pipeline.html"
    )


@app.route("/dataset")
def dataset():

    return render_template(
        "dataset.html"
    )


@app.route("/model")
def model():

    return render_template(
        "model.html"
    )


@app.route("/inference")
def inference():

    return render_template(
        "inference.html"
    )


# ==========================================================
# Uploaded Image
# ==========================================================

@app.route("/uploaded")
def uploaded():

    global LATEST_FILE

    image_path = os.path.join(
        BASE_DIR,
        "uploads",
        LATEST_FILE
    )

    print(f"[UPLOAD IMAGE] {image_path}")

    return send_file(image_path)


# ==========================================================
# Enhanced Image
# ==========================================================

@app.route("/enhanced")
def enhanced():

    image_path = os.path.join(
        RESULT_FOLDER,
        "enhanced.jpg"
    )

    print(f"[ENHANCED IMAGE] {image_path}")

    return send_file(
        image_path,
        mimetype="image/jpeg"
    )


# ==========================================================
# Upload & AI Inference
# ==========================================================

@app.route("/upload", methods=["POST"])
def upload():

    print("\n" + "=" * 70)
    print("UPLOAD ROUTE EXECUTED")
    print("=" * 70)

    global LATEST_FILE
    global LATEST_REPORT

    file = request.files.get("image")

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    if file is None:
        print("ERROR : No file uploaded.")
        return "No file uploaded."

    if file.filename == "":
        print("ERROR : Empty filename.")
        return "No file selected."

    if not allowed_file(file.filename):
        print("ERROR : Unsupported file type.")
        return "Unsupported image format."

    print("Selected File :", file.filename)

    # ------------------------------------------------------
    # Save Uploaded Image
    # ------------------------------------------------------

    filename = secure_filename(file.filename)
    LATEST_FILE = filename

    upload_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    file.save(upload_path)

    print("Saved Upload :", upload_path)

    # ------------------------------------------------------
    # Output Path
    # ------------------------------------------------------

    enhanced_path = os.path.join(
        RESULT_FOLDER,
        "enhanced.jpg"
    )

    print("Output Image :", enhanced_path)

    # ------------------------------------------------------
    # AI Inference
    # ------------------------------------------------------

    try:

        print("Calling enhance_infrared()...")

        report = enhance_infrared(
            upload_path,
            enhanced_path
        )

        print("Inference Completed Successfully")

        LATEST_REPORT = report

        print("Report Keys :", list(report.keys()))
        print("=" * 70)

        return render_template(
            "result.html",
            report=report
        )

    except Exception as e:

        import traceback

        print("\nERROR DURING INFERENCE")
        traceback.print_exc()

        return f"""
        <h2>Processing Failed</h2>
        <pre>{traceback.format_exc()}</pre>
        """
@app.route("/download-report")
def download_report():

    if not LATEST_REPORT:
        return "No report available. Please process an image first."

    pdf_path = os.path.join(
        RESULT_FOLDER,
        "VoidHunter_AI_Report.pdf"
    )
    os.makedirs(
    RESULT_FOLDER,
    exist_ok=True
)

    doc = SimpleDocTemplate(

    pdf_path,

    rightMargin=30,

    leftMargin=30,

    topMargin=45,

    bottomMargin=45

)


    styles = getSampleStyleSheet()

    elements = []

    from reportlab.lib.enums import TA_CENTER

    title_style = styles["Title"]
    title_style.alignment = TA_CENTER

    heading_style = styles["Heading2"]
    heading_style.alignment = TA_CENTER

    normal_style = styles["BodyText"]
    normal_style.alignment = TA_CENTER

    # ======================================================
    # Title
    # ======================================================

    elements.append(

        Paragraph(
            "<font size='30' color='#0B5ED7'><b>VOID HUNTER</b></font>",
            title_style
        )

    )

    elements.append(

      Paragraph(
          
            "<b>AI Surface Analysis Report</b>",
            heading_style

        )

    )

    elements.append(

        Paragraph(

            f"Generated : {datetime.now().strftime('%d %B %Y  %H:%M:%S')}",

            normal_style

        )

    )

    elements.append(

        Spacer(1,20)

    )# ======================================================
    # Images
    # ======================================================

    original_image = os.path.join(
        BASE_DIR,
        "uploads",
        LATEST_FILE
    )

    enhanced_image = os.path.join(
        BASE_DIR,
        "static",
        "results",
        "enhanced.jpg"
    )

    elements.append(
        Paragraph(
            "<b>Image Comparison</b>",
            heading_style
        )
    )

    image_data = []

    left_image = None
    right_image = None

    if os.path.exists(original_image):

        left_image = Image(
            original_image,
            width=3*inch,
            height=3*inch
        )

    if os.path.exists(enhanced_image):

        right_image = Image(
            enhanced_image,
            width=3*inch,
            height=3*inch
        )

    image_data = [

    [

        Paragraph(
            "<b>Original Infrared Image</b>",
            normal_style
        ),

        Paragraph(
            "<b>Enhanced RGB Output</b>",
            normal_style
        )

    ],

    [

        left_image,

        right_image

    ]

]

    image_table = Table(
        image_data,
        colWidths=[3*inch,3*inch]
    )

    image_table.setStyle(

    TableStyle([

        ("ALIGN",(0,0),(-1,-1),"CENTER"),

        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),

        ("BOTTOMPADDING",(0,0),(-1,-1),12),

        ("TOPPADDING",(0,0),(-1,-1),8),

        ("LEFTPADDING",(0,0),(-1,-1),8),

        ("RIGHTPADDING",(0,0),(-1,-1),8),

    ])

)

    elements.append(
        image_table
    )

    elements.append(
        Spacer(1,20)
    )

    # ======================================================
    # Processing Information
    # ======================================================

    elements.append(

        Paragraph(

            "<b>Processing Information</b>",

            heading_style

        )

    )
    
    processing_data = [

    ["Parameter", "Value"],

    ["Status", LATEST_REPORT.get("status", "N/A")],

    ["Inference Time", LATEST_REPORT.get("inference_time", "N/A")],

    ["Processing Device", LATEST_REPORT.get("device", "N/A")],

    ["Model Name", LATEST_REPORT.get("model_name", "N/A")],

    ["Framework", LATEST_REPORT.get("framework", "N/A")],

    ["Prediction", LATEST_REPORT.get("prediction", "N/A")],

    ["Parameters", LATEST_REPORT.get("parameters", "N/A")]

]

    processing_table = Table(
        processing_data,
        colWidths=[2.8*inch, 3.2*inch]
    )

    processing_table.setStyle(

        TableStyle([

            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#003366")),

            ("TEXTCOLOR", (0,0), (-1,0), colors.white),

            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),

            ("FONTSIZE", (0,0), (-1,-1), 10),

            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),

            ("BOTTOMPADDING", (0,0), (-1,0), 10),

            ("BACKGROUND", (0,1), (-1,-1), colors.whitesmoke),

            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),

            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("LEFTPADDING",(0,0),(-1,-1),8),

            ("RIGHTPADDING",(0,0),(-1,-1),8),

            ("TOPPADDING",(0,0),(-1,-1),8),

            ("BOTTOMPADDING",(0,0),(-1,-1),8),

        ])

    )

    elements.append(
        processing_table
    )
    elements.append(
    Spacer(1,12)
)


    image_info = [

        ["Parameter", "Value"],

        ["Input Resolution", LATEST_REPORT.get("input_resolution", "N/A")],

        ["Output Resolution", LATEST_REPORT.get("output_resolution", "N/A")],

        ["Input Size", LATEST_REPORT.get("input_size", "N/A")],

        ["Output Size", LATEST_REPORT.get("output_size", "N/A")]

    ]

    image_info_table = Table(
        image_info,
        colWidths=[2.8*inch,3.2*inch]
    )
    image_info_table.setStyle(

        TableStyle([

            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#003366")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1),9),
            ("GRID",(0,0),(-1,-1),0.5,colors.grey),
            ("BACKGROUND",(0,1),(-1,-1),colors.whitesmoke),
            ("ALIGN",(0,0),(-1,-1),"CENTER"),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE")

        ])

    )

    elements.append(

    KeepTogether([

        Paragraph(
            "<b>Image Information</b>",
            heading_style
        ),

        Spacer(1, 6),

        image_info_table

    ])

)

    elements.append(
    Spacer(1,10)
    )

    # ======================================================
    # AI Evaluation Metrics
    # ======================================================

    elements.append(

        Paragraph(

            "<b>AI Evaluation Metrics</b>",

            heading_style

        )

    )

    metric_data = [

        ["Metric", "Value"],

        ["PSNR", LATEST_REPORT.get("psnr", "N/A")],

        ["SSIM", LATEST_REPORT.get("ssim", "N/A")],

        ["MSE", LATEST_REPORT.get("mse", "N/A")],

        ["LPIPS", LATEST_REPORT.get("lpips", "N/A")],

        ["Training Loss", LATEST_REPORT.get("training_loss", "N/A")],

        ["Validation Loss", LATEST_REPORT.get("validation_loss", "N/A")],

        ["Ground Truth", LATEST_REPORT.get("ground_truth", "N/A")],

        ["Evaluation Status", LATEST_REPORT.get("evaluation_status", "N/A")]

    ]

    metric_table = Table(
        metric_data,
        colWidths=[2.8*inch, 3.2*inch]
    )

    metric_table.setStyle(

        TableStyle([

            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#198754")),

            ("TEXTCOLOR", (0,0), (-1,0), colors.white),

            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),

            ("FONTSIZE", (0,0), (-1,-1), 10),

            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),

            ("BOTTOMPADDING", (0,0), (-1,0), 10),

            ("BACKGROUND", (0,1), (-1,-1), colors.beige),

            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),

            ("ALIGN", (0,0), (-1,-1), "CENTER"),

            ("LEFTPADDING",(0,0),(-1,-1),8),

            ("RIGHTPADDING",(0,0),(-1,-1),8),

            ("TOPPADDING",(0,0),(-1,-1),8),

            ("BOTTOMPADDING",(0,0),(-1,-1),8),

        ])

    )

    elements.append(
        metric_table
    )

    elements.append(
        Spacer(1,20)
    )
    # ======================================================
    # Footer
    # ======================================================

    elements.append(
        Spacer(1,20)
    )

    elements.append(

        Paragraph(

            "<font size='14'><b>Team VOID HUNTER</b></font>",

            heading_style

        )

    )

    elements.append(

        Paragraph(

            "AI-Based Infrared Image Enhancement System",

            normal_style

        )

    )

    elements.append(

        Paragraph(

            "Developed for Smart India Hackathon (SIH) 2026",

            normal_style

        )

    )

    elements.append(

        Paragraph(

            f"Generated on : {datetime.now().strftime('%d %B %Y %H:%M:%S')}",

            normal_style

        )

    )
    # ======================================================
    # Build PDF
    # ======================================================

    doc.build(elements)

    return send_file(

        pdf_path,

        as_attachment=True,

        download_name="VoidHunter_AI_Report.pdf",

        mimetype="application/pdf"

    )
# Main
# ==========================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )
