from flask import Flask, request, jsonify, Blueprint
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import pydicom

x_ray_bp = Blueprint('x_ray_analyze', __name__, url_prefix='/api')


UPLOAD_FOLDER = 'uploads/classification'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def classify_image_with_cv(img):
    """
    Heuristic classification of medical X-ray image using OpenCV features.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return "Unknown"

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = float(w) / h
    extent = area / (w * h)

    print(f"[DEBUG] Area: {area}, AR: {aspect_ratio:.2f}, Extent: {extent:.2f}")

    # ---------------------------- CLASSIFICATION RULES ---------------------------- #

   # Skull or brain-related
    if area > 10000 and 0.9 <= aspect_ratio <= 1.2 and extent > 0.5:
        return "Head / Brain"

    # Sinus
    if 8000 < area < 15000 and 0.8 <= aspect_ratio <= 1.2 and extent > 0.5:
        return "Sinus"

    # Chest/Thorax
    if area > 25000 and 0.7 <= aspect_ratio <= 1.5 and extent > 0.4:
        return "Chest / Thorax"
    # if 25000 < area < 200000 and 0.7 <= aspect_ratio <= 1.5 and extent > 0.4:
    #     return "Chest / Thorax"

    # Abdomen
    if area > 18000 and 0.8 <= aspect_ratio <= 1.3 and extent > 0.4:
        return "Abdomen / Pelvis"

    # Spine (Cervical, Thoracic, Lumbar)
    if area > 10000 and aspect_ratio < 0.4 and extent < 0.5:
        return "Spine"

    # Shoulder or Clavicle
    if 7000 < area < 15000 and 1.2 <= aspect_ratio <= 2.0 and extent > 0.4:
        return "Shoulder / Clavicle"

    # Upper limb (Hand, Arm, Elbow)
    if area > 5000 and aspect_ratio < 1.0 and extent > 0.3:
        return "Upper Limb (Arm / Elbow / Hand)"

    # Lower limb (Leg, Knee, Foot)
    if area > 12000 and (0.6 <= aspect_ratio <= 1.2 or aspect_ratio > 2.0):
        return "Lower Limb (Leg / Knee / Foot)"

    # Dental (Teeth/Jaw)
    if area < 8000 and aspect_ratio > 2.0:
        return "Dental"

    # Pelvis/Hip
    if area > 15000 and 0.85 <= aspect_ratio <= 1.4 and extent > 0.45:
        return "Pelvis / Hip"

    # Contrast/Special procedures (e.g., angiography)
    if area > 10000 and extent < 0.3:
        return "Contrast / Special Studies"

    # Miscellaneous
    if area > 8000:
        return "Other Body Region"

    return "Unclear / Other"

def extract_body_part(dicom_path):
    try:
        ds = pydicom.dcmread(dicom_path)

        def get_str(tag):
            val = ds.get(tag)
            return val.value.strip() if val and isinstance(val.value, str) else ''

        # Try known metadata fields
        body_part = get_str((0x0018, 0x0015))
        acquisition_desc = get_str((0x0018, 0x1400))
        study_description = get_str((0x0008, 0x1030))
        series_description = get_str((0x0008, 0x103E))

        candidates = [body_part, acquisition_desc, study_description, series_description]
        candidates = [c for c in candidates if c]  # Filter out empty values

        if not candidates:
            return "Unknown / Not Specified"

        # Normalize and classify
        for c in candidates:
            normalized = c.upper()
            if "WRIST" in normalized or "FOREARM" in normalized:
                return "Upper Limb (Wrist / Forearm)"
            elif "SHOULDER" in normalized:
                return "Shoulder / Clavicle"
            elif "CHEST" in normalized or "THORAX" in normalized:
                return "Chest / Thorax"
            elif "SPINE" in normalized:
                return "Spine"
            elif "HEAD" in normalized or "BRAIN" in normalized or "SKULL" in normalized:
                return "Head / Brain"
            elif "KNEE" in normalized or "LEG" in normalized:
                return "Lower Limb (Knee / Leg)"
            elif "ABDOMEN" in normalized or "PELVIS" in normalized:
                return "Abdomen / Pelvis"
            elif "DENTAL" in normalized or "JAW" in normalized:
                return "Dental"
            # else:
            #     return "NaN"

        return candidates[0]  # Return first available if no pattern matched

    except Exception as e:
        return f"Error: {str(e)}"

    
@x_ray_bp.route('/analyse_pic', methods=['POST'])
def analyse_pic():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ['.jpg', '.jpeg', '.png', '.dcm']:
        return jsonify({"error": "Unsupported file type"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Load image using OpenCV or pydicom
    if ext == ".dcm":
        dicom_data = pydicom.dcmread(filepath)
        classification = extract_body_part(filepath)
        img = dicom_data.pixel_array
        img = cv2.convertScaleAbs(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


    else:
        img = cv2.imread(filepath)
        if len(img.shape) == 3:
            b, g, r = cv2.split(img)
            
            if not (np.array_equal(b, g) and np.array_equal(g, r)):
                return jsonify({"error": "Colored images are not supported"}), 400
            classification = classify_image_with_cv(img)

    if img is None:
        return jsonify({"error": "Failed to read image"}), 500

    

    print(f"Classified Image as: {classification}")

    return jsonify({"classification": classification})

 # Head/Brain
    # if area > 15000 and 0.9 <= aspect_ratio <= 1.2:
    #     return "Skull"
    # elif area > 12000 and aspect_ratio > 1.3 and extent < 0.4:
    #     return "Sinus"
    # elif area > 10000 and extent < 0.3:
    #     return "Cerebral Angio"
    # # Dental
    # if area < 6000 and aspect_ratio > 2.5:
    #     return "Bitewing"
    # elif area < 5000 and aspect_ratio < 1.0:
    #     return "Periapical"
    # elif 6000 <= area <= 15000 and aspect_ratio > 3.0:
    #     return "Panoramic"
    # elif area < 8000 and 1.0 <= aspect_ratio <= 1.5:
    #     return "Cephalometric"
    # # Chest/Thorax
    # if area > 25000 and 0.7 <= aspect_ratio <= 1.3 and extent > 0.5:
    #     return "CXR"
    # elif area > 10000 and aspect_ratio > 2.5:
    #     return "Rib"
    # elif area > 9000 and aspect_ratio < 0.5:
    #     return "Clavicle"
    # elif area > 12000 and aspect_ratio < 0.4:
    #     return "Thoracic Spine"
    # # spine
    # if area > 10000 and 0.3 <= aspect_ratio <= 0.5:
    #     return "Cervical"
    # elif area > 12000 and 0.3 <= aspect_ratio <= 0.5:
    #     return "Thoracic"
    # elif area > 14000 and 0.3 <= aspect_ratio <= 0.5:
    #     return "Lumbar"
    # elif area > 8000 and aspect_ratio < 0.4:
    #     return "Sacrum/Coccyx"
    # elif area > 15000 and aspect_ratio < 0.3:
    #     return "Scoliosis"
    # # Upper limb
    # if area > 10000 and 0.7 <= aspect_ratio <= 1.3:
    #     return "Shoulder"
    # elif area > 8000 and aspect_ratio < 0.6:
    #     return "Arm"
    # elif area > 7000 and aspect_ratio < 0.5:
    #     return "Elbow"
    # elif area > 6000 and aspect_ratio < 0.5:
    #     return "Forearm"
    # elif area > 4000 and aspect_ratio < 0.6:
    #     return "Wrist"
    # elif area > 3000 and aspect_ratio < 0.5:
    #     return "Hand"
    # elif area < 3000 and aspect_ratio < 0.4:
    #     return "Fingers"
    # # Lower Limb
    # if area > 15000 and 1.0 <= aspect_ratio <= 1.5:
    #     return "Pelvis"
    # elif area > 14000 and aspect_ratio > 2.0:
    #     return "Hip"
    # elif area > 13000 and aspect_ratio < 0.5:
    #     return "Femur"
    # elif area > 11000 and aspect_ratio < 0.5:
    #     return "Knee"
    # elif area > 10000 and aspect_ratio < 0.5:
    #     return "Tibia/Fibula"
    # elif area > 5000 and 0.7 <= aspect_ratio <= 1.2:
    #     return "Ankle"
    # elif area > 3000 and aspect_ratio < 0.6:
    #     return "Foot"
    # elif area < 3000 and aspect_ratio < 0.4:
    #     return "Toes"
    # # Abdomen
    # if area > 20000 and 0.8 <= aspect_ratio <= 1.3:
    #     return "KUB"
    # elif area > 22000 and 0.9 <= aspect_ratio <= 1.4:
    #     return "Abdominal X-ray"
    # elif area > 20000 and 0.8 <= aspect_ratio <= 1.5:
    #     return "Pelvic X-ray"
    # elif area > 18000 and extent < 0.4:
    #     return "IVP"
    # # Contract special
    # if area > 15000 and extent < 0.3:
    #     return "Barium Studies"
    # elif area > 18000 and 0.8 <= aspect_ratio <= 1.2:
    #     return "Mammogram"
    # elif area > 10000 and aspect_ratio < 0.5:
    #     return "HSG"
    # elif area > 12000 and aspect_ratio < 0.6:
    #     return "Myelogram"
    # elif area > 10000 and extent < 0.2:
    #     return "DEXA"
    # elif area > 9000 and extent < 0.25:
    #     return "Angiography"
# from flask import Flask, request, jsonify, Blueprint
# import os
# import cv2
# import numpy as np
# from werkzeug.utils import secure_filename
# import pydicom
 
# x_ray_bp = Blueprint('x_ray_analyze', __name__, url_prefix='/api')
 
# UPLOAD_FOLDER = 'uploads/classification'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
 
# def classify_image_with_cv(img):
#     """Heuristic classification of medical X-ray image using OpenCV features."""
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
#     if len(contours) == 0:
#         return "Unclear / Other"
 
#     largest_contour = max(contours, key=cv2.contourArea)
#     area = cv2.contourArea(largest_contour)
#     x, y, w, h = cv2.boundingRect(largest_contour)
#     aspect_ratio = float(w) / h
#     extent = area / (w * h)
 
#     # ---------------------------- CLASSIFICATION RULES ---------------------------- #
#     if area > 10000 and 0.9 <= aspect_ratio <= 1.2 and extent > 0.5:
#         return "Head / Brain"
#     if 8000 < area < 15000 and 0.8 <= aspect_ratio <= 1.2 and extent > 0.5:
#         return "Sinus"
#     if area > 25000 and 0.7 <= aspect_ratio <= 1.5 and extent > 0.4:
#         return "Chest / Thorax"
#     if area > 18000 and 0.9 <= aspect_ratio <= 1.3 and extent > 0.4:
#         return "Abdomen / Pelvis"
#     if area > 10000 and aspect_ratio < 0.4 and extent < 0.5:
#         return "Spine"
#     if 7000 < area < 15000 and 1.2 <= aspect_ratio <= 2.0 and extent > 0.4:
#         return "Shoulder / Clavicle"
#     if area > 5000 and aspect_ratio < 1.0 and extent > 0.3:
#         return "Upper Limb (Arm / Elbow / Hand)"
#     if area > 12000 and (0.6 <= aspect_ratio <= 1.2 or aspect_ratio > 2.0):
#         return "Lower Limb (Leg / Knee / Foot)"
#     if area < 8000 and aspect_ratio > 2.0:
#         return "Dental"
#     if area > 15000 and 1.0 <= aspect_ratio <= 1.4 and extent > 0.45:
#         return "Pelvis / Hip"
#     if area > 10000 and extent < 0.3:
#         return "Contrast / Special Studies"
#     if area > 8000:
#         return "Other Body Region"
 
#     return "Unclear / Other"
 
# @x_ray_bp.route('/analyse_pic', methods=['POST'])
# def analyse_pic():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file provided"}), 400
 
#     file = request.files['file']
#     filename = secure_filename(file.filename)
#     ext = os.path.splitext(filename)[1].lower()
 
#     if ext not in ['.jpg', '.jpeg', '.png', '.dcm']:
#         return jsonify({"error": "Unsupported file type"}), 400
 
#     filepath = os.path.join(UPLOAD_FOLDER, filename)
#     file.save(filepath)
 
#     try:
#         if ext == ".dcm":
#             dicom_data = pydicom.dcmread(filepath)
#             if not hasattr(dicom_data, 'pixel_array'):
#                 return jsonify({"error": "DICOM file does not contain image data"}), 400
 
#             # Check modality
#             modality = dicom_data.get("Modality", "").upper()
#             if modality not in ["CR", "DX", "MR", "CT", "XRAY"]:
#                 return jsonify({"error": f"Unsupported DICOM modality: {modality}"}), 400
 
#             img = dicom_data.pixel_array
#             img = cv2.convertScaleAbs(img)
#             img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#         else:
#             img = cv2.imread(filepath)
#             if img is None:
#                 return jsonify({"error": "Failed to read image"}), 500
 
#         classification = classify_image_with_cv(img)
 
#         if classification == "Unclear / Other":
#             return jsonify({"error": "Image unclear or not a valid X-ray"}), 400
 
#         return jsonify({"classification": classification})
 
#     except Exception as e:
#         return jsonify({"error": f"Processing error: {str(e)}"}), 500
# from flask import Blueprint, request, jsonify
# from werkzeug.utils import secure_filename
# import os, cv2, numpy as np, pydicom, scipy.signal as sig

# x_ray_bp = Blueprint("x_ray_analyze", __name__, url_prefix="/api")
# UPLOAD_FOLDER = "uploads/classification"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # ---------------------------- TUNEABLE CONSTANTS -----------------------------
# STD_MIN        = 20          # ≥20 pixel st‑dev  → sufficient contrast  :contentReference[oaicite:0]{index=0}
# EDGE_MIN       = 0.025       # ≥2.5 % Canny edges → bone‑rich image    :contentReference[oaicite:1]{index=1}
# GRAYSCALE_VAR  = 50          # RGB variance ≤50   → “near‑monochrome”
# PEAKS_REQUIRED = 2           # ≥2 histogram peaks → bi‑modal X‑ray look

# # Area/aspect limits for anatomy rules (fractions of image area)
# CHEST_RATIO    = (0.40, 0.75)
# PELVIS_RATIO   = (0.30, 0.60)
# SPINE_RATIO    = (0.10, 0.30)
# LIMB_RATIO     = (0.05, 0.25)

# def _ensure_gray(img):
#     if img.ndim == 3:
#         var_rgb = np.mean((img[:,:,0]-img[:,:,1])**2 +
#                           (img[:,:,1]-img[:,:,2])**2 +
#                           (img[:,:,2]-img[:,:,0])**2)
#         if var_rgb > GRAYSCALE_VAR:
#             return None          # probably NOT an X‑ray
#         return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     return img.copy()

# def is_probable_xray(gray):
#     # 1️⃣ contrast
#     if gray.std() < STD_MIN:
#         return False
#     # 2️⃣ edge density
#     edges = cv2.Canny(gray, 50, 150)
#     if edges.mean() < EDGE_MIN:
#         return False
#     # 3️⃣ histogram peaks
#     hist = cv2.calcHist([gray],[0],None,[256],[0,256]).flatten()
#     peaks,_ = sig.find_peaks(hist, height=hist.max()*0.05, distance=10)
#     if len(peaks) < PEAKS_REQUIRED:
#         return False
#     return True

# def classify_bodypart(gray):
#     # 1. equalise → threshold
#     eq   = cv2.equalizeHist(gray)
#     _,th = cv2.threshold(eq, 0, 255, cv2.THRESH_OTSU)

#     # 2. close gaps so both lungs merge
#     close_kernel = np.ones((int(0.03*gray.shape[0]),) * 2, np.uint8)
#     th  = cv2.morphologyEx(th, cv2.MORPH_CLOSE, close_kernel, iterations=2)

#     contours,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         return "Unclear"

#     cnt   = max(contours, key=cv2.contourArea)
#     area  = cv2.contourArea(cnt)
#     h,w   = gray.shape
#     ratio = area / (h*w)
#     x,y,bw,bh = cv2.boundingRect(cnt)
#     ar    = bw / bh                       # aspect ratio

#     # ---------- tuned rules ----------
#     if 0.25 <= ratio <= 0.80 and 0.55 <= ar <= 1.40 and y < h*0.25:
#         return "Chest / Thorax"
#     if 0.28 <= ratio <= 0.60 and 0.80 <= ar <= 1.30 and y > h*0.35:
#         return "Pelvis / Abdomen"
#     if 0.08 <= ratio <= 0.30 and ar < 0.50 and bh > bw*1.5:
#         return "Spine (AP/Lateral)"
#     if ratio <= 0.25 and ar > 1.5:
#         return "Upper / Lower Limb"
#     if ar > 2.5 and ratio < 0.08:
#         return "Dental / Mandible"
#     return "Other / Unclear"

# # ---------------------------- FLASK ENDPOINT ---------------------------------
# @x_ray_bp.route("/analyse_pic", methods=["POST"])
# def analyse_pic():
#     if "file" not in request.files:
#         return jsonify(error="No file provided"), 400
#     f       = request.files["file"]
#     fname   = secure_filename(f.filename)
#     ext     = os.path.splitext(fname)[1].lower()
#     if ext not in {".jpg",".jpeg",".png",".dcm"}:
#         return jsonify(error="Unsupported file type"), 400

#     path = os.path.join(UPLOAD_FOLDER, fname)
#     f.save(path)

#     # --- LOAD PIXELS ---------------------------------------------------------
#     try:
#         if ext == ".dcm":
#             ds = pydicom.dcmread(path)
#             if ds.get("Modality","").upper() not in {"CR","DX","DR"}:
#                 return jsonify(error="Not a planar X‑ray DICOM"), 400
#             pix  = cv2.convertScaleAbs(ds.pixel_array)
#         else:
#             pix  = cv2.imread(path, cv2.IMREAD_UNCHANGED)
#             if pix is None:
#                 return jsonify(error="Image decode failed"), 500
#     except Exception as e:
#         return jsonify(error=f"Read error: {e}"), 500

#     gray = _ensure_gray(pix)
#     if gray is None:
#         return jsonify(is_xray=False,
#                        reason="Image is coloured / not radiograph"), 200

#     if not is_probable_xray(gray):
#         return jsonify(is_xray=False,
#                        reason="Failed heuristic X‑ray checks"), 200

#     # ----------------- body‑part rules -----------------
#     label = classify_bodypart(gray)
#     print(label)

#     if label == "Other / Unclear":
#         return jsonify(is_xray=True,
#                        classification=None,
#                        note="Anatomy not recognised"), 200
#     return jsonify(is_xray=True,
#                    classification=label), 200
# # x_ray_analyze_rules.py  – DROP‑IN REPLACEMENT
# from flask import Blueprint, request, jsonify
# from werkzeug.utils import secure_filename
# import os, cv2, numpy as np, pydicom

# x_ray_bp = Blueprint("x_ray_analyze", __name__, url_prefix="/api")
# UPLOAD_FOLDER = "uploads/classification"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # ---------------------------- TUNE HERE --------------------------------------
# # area‑ratio windows are fractions of the whole frame (height×width)
# CHEST_RATIO   = (0.35, 0.85)   # covers most PA/AP films
# ABDO_RATIO    = (0.25, 0.60)
# SPINE_RATIO   = (0.05, 0.18)
# LIMB_RATIO    = 0.30           # “smallish” blobs
# # morphological kernel is 3 % of min(image‑side)
# CLOSE_FRAC    = 0.03
# # ---------------------------------------------------------------------------

# def _load_pixels(path, ext):
#     if ext == ".dcm":
#         ds  = pydicom.dcmread(path)
#         img = cv2.convertScaleAbs(ds.pixel_array)
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#         return img
#     img = cv2.imread(path)
#     return img

# def classify_bodypart(img):
#     """Heuristic anatomy classifier – NO ML."""
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     eq   = cv2.equalizeHist(gray)

#     # 1️⃣  Otsu threshold  ▸  2️⃣ invert so lungs/body are white
#     _, bin_img = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     fg = cv2.bitwise_not(bin_img)

#     # 3️⃣  Close gaps so the two hemithoraces become one blob
#     h, w = gray.shape
#     k    = max(5, int(min(h, w) * CLOSE_FRAC))
#     kernel = np.ones((k, k), np.uint8)
#     fg  = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)

#     # 4️⃣  Largest contour → geometry features
#     cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not cnts:
#         return "Unclear / Other"

#     cnt   = max(cnts, key=cv2.contourArea)
#     area  = cv2.contourArea(cnt)
#     ratio = area / (h * w)               # % of frame occupied
#     x, y, bw, bh = cv2.boundingRect(cnt) # bounding box
#     ar    = bw / bh                      # width / height

#     # ---------------------------- RULES -------------------------------------
#     if CHEST_RATIO[0] <= ratio <= CHEST_RATIO[1] and 0.60 <= ar <= 1.50 and y < h * 0.25:
#         return "Chest / Thorax"

#     if ABDO_RATIO[0] <= ratio <= ABDO_RATIO[1] and 0.80 <= ar <= 1.30 and y > h * 0.30:
#         return "Abdomen / Pelvis"

#     if SPINE_RATIO[0] <= ratio <= SPINE_RATIO[1] and ar < 0.50 and bh > bw * 1.5:
#         return "Spine"

#     if ratio < LIMB_RATIO and ar > 1.6:
#         return "Upper / Lower Limb"

#     if ar > 2.5 and ratio < 0.10:
#         return "Dental"

#     return "Other / Unclear"

# # ---------------------------------------------------------------------------

# @x_ray_bp.route("/analyse_pic", methods=["POST"])
# def analyse_pic():
#     if "file" not in request.files:
#         return jsonify(error="No file provided"), 400

#     f     = request.files["file"]
#     fname = secure_filename(f.filename)
#     ext   = os.path.splitext(fname)[1].lower()

#     if ext not in {".jpg", ".jpeg", ".png", ".dcm"}:
#         return jsonify(error="Unsupported file type"), 400

#     path = os.path.join(UPLOAD_FOLDER, fname)
#     f.save(path)

#     img = _load_pixels(path, ext)
#     if img is None:
#         return jsonify(error="Failed to decode image"), 500

#     label = classify_bodypart(img)
#     return jsonify(classification=label), 200

# from flask import Blueprint, request, jsonify
# from werkzeug.utils import secure_filename
# import os, cv2, numpy as np, pydicom, scipy.signal as sig

# x_ray_bp = Blueprint("x_ray_analyze", __name__, url_prefix="/api")
# UPLOAD_FOLDER = "uploads/classification"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # ---------- GLOBAL CONSTANTS (tune if needed) -------------------------------
# STD_MIN        = 18          # contrast floor (σ of gray levels)
# EDGE_MIN       = 0.020       # ≥2 % Canny edges → bone‑rich
# GRAYSCALE_VAR  = 45          # RGB variance ceiling → near‑monochrome
# PEAKS_REQ      = 2           # ≥2 histogram peaks
# CLOSE_FRAC     = 0.03        # % of min(H,W) for closing kernel

# # area‑ratio windows
# CHEST_R        = (0.35, 0.85)
# PELVIS_R       = (0.30, 0.65)
# SPINE_R        = (0.05, 0.18)
# LIMB_MAX_R     = 0.35        # anything smaller than this might be a limb

# # ---------------------------------------------------------------------------

# def load_pixels(path, ext):
#     if ext == ".dcm":
#         ds = pydicom.dcmread(path)
#         if not hasattr(ds, "pixel_array"):
#             raise ValueError("DICOM contains no image data")
#         img = cv2.convertScaleAbs(ds.pixel_array)
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     else:
#         img = cv2.imread(path)
#         if img is None:
#             raise ValueError("Image decode failed")
#     return img

# # ---------- BASIC X‑RAY VERIFICATION ----------------------------------------
# def looks_like_xray(bgr):
#     # 1. reject strong colour content
#     var_rgb = np.mean((bgr[:,:,0]-bgr[:,:,1])**2 +
#                       (bgr[:,:,1]-bgr[:,:,2])**2 +
#                       (bgr[:,:,2]-bgr[:,:,0])**2)
#     if var_rgb > GRAYSCALE_VAR:
#         return False, "Image is coloured / not radiograph"

#     gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
#     if gray.std() < STD_MIN:
#         return False, "Low‑contrast image"

#     # edge density
#     edges = cv2.Canny(gray, 50, 150)
#     if edges.mean() < EDGE_MIN:
#         return False, "Too few bone edges for an X‑ray"

#     # bi‑modal histogram (bone + soft tissue)
#     hist = cv2.calcHist([gray],[0],None,[256],[0,256]).flatten()
#     peaks, _ = sig.find_peaks(hist, height=hist.max()*0.05, distance=10)
#     if len(peaks) < PEAKS_REQ:
#         return False, "Histogram not bi‑modal"

#     return True, gray

# # ---------- BODY‑PART HEURISTICS -------------------------------------------
# def classify_bodypart(gray):
#     # threshold & invert
#     _, th = cv2.threshold(cv2.equalizeHist(gray),
#                           0, 255,
#                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     th = cv2.bitwise_not(th)

#     # close so lungs merge
#     k = max(5, int(min(gray.shape) * CLOSE_FRAC))
#     th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
#                           np.ones((k,k), np.uint8), iterations=2)

#     cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL,
#                                cv2.CHAIN_APPROX_SIMPLE)
#     if not cnts:
#         return "Unclear / Other"

#     cnt   = max(cnts, key=cv2.contourArea)
#     area  = cv2.contourArea(cnt)
#     h, w  = gray.shape
#     ratio = area / (h * w)
#     x, y, bw, bh = cv2.boundingRect(cnt)
#     ar    = bw / bh                # width / height

#     # Chest
#     if CHEST_R[0] <= ratio <= CHEST_R[1] and \
#        0.60 <= ar <= 1.50 and y < h * 0.18:
#         return "Chest / Thorax"

#     # Pelvis / Abdomen
#     if PELVIS_R[0] <= ratio <= PELVIS_R[1] and \
#        0.55 <= ar <= 1.25 and y >= h * 0.18:
#         return "Pelvis / Abdomen"

#     # Spine
#     if SPINE_R[0] <= ratio <= SPINE_R[1] and ar < 0.50 and bh > bw * 1.5:
#         return "Spine"

#     # Limbs  – portrait OR landscape
#     if ratio < LIMB_MAX_R and (ar < 0.65 or ar > 1.60):
#         return "Upper / Lower Limb"

#     # Dental
#     if ar > 2.5 and ratio < 0.10:
#         return "Dental / Mandible"

#     return "Other / Unclear"

# # ---------- FLASK ENDPOINT --------------------------------------------------
# @x_ray_bp.route("/analyse_pic", methods=["POST"])
# def analyse_pic():
#     if "file" not in request.files:
#         return jsonify(error="No file provided"), 400

#     f     = request.files["file"]
#     fname = secure_filename(f.filename)
#     ext   = os.path.splitext(fname)[1].lower()
#     if ext not in {".jpg", ".jpeg", ".png", ".dcm"}:
#         return jsonify(error="Unsupported file type"), 400

#     path = os.path.join(UPLOAD_FOLDER, fname)
#     f.save(path)

#     try:
#         bgr = load_pixels(path, ext)
#     except ValueError as e:
#         return jsonify(error=str(e)), 400

#     ok, out = looks_like_xray(bgr)
#     if not ok:
#         return jsonify(is_xray=False, reason=out), 200

#     label = classify_bodypart(out)
#     print(label)
#     return jsonify(is_xray=True, classification=label), 200