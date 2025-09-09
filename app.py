from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import cv2
import re
import google.generativeai as genai
from werkzeug.utils import secure_filename
import base64
from PIL import Image, ImageOps
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Optional HEIC/HEIF support
try:
    from pillow_heif import register_heif_opener  # type: ignore
    register_heif_opener()
    HEIF_SUPPORTED = True
except Exception:
    HEIF_SUPPORTED = False

# Allowed file extensions (include webp and heic/heif if supported)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
if HEIF_SUPPORTED:
    ALLOWED_EXTENSIONS.update({'heic', 'heif'})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(src_path: str, out_dir: str) -> str:
    """
    Preprocess image to improve OCR on compressed WhatsApp images:
    - Fix EXIF orientation
    - Convert to grayscale
    - Optional upscale for small images
    - Denoise and adaptive threshold
    Returns path to processed image.
    """
    try:
        # Load via PIL to handle formats/EXIF cleanly
        with Image.open(src_path) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img)
            pil_img = pil_img.convert('RGB')

            width, height = pil_img.size
            if width < 800:
                scale = min(2.0, 800 / max(1, width))
                new_size = (int(width * scale), int(height * scale))
                pil_img = pil_img.resize(new_size, Image.LANCZOS)

            # Convert to numpy for OpenCV processing
            img_bytes = io.BytesIO()
            pil_img.save(img_bytes, format='JPEG', quality=95)
            img_bytes.seek(0)
            data = img_bytes.read()

        # OpenCV processing (enhanced for OCR)
        import numpy as np
        np_arr = np.frombuffer(data, np.uint8)
        cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # 1) Grayscale
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # 2) Contrast enhancement (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # 3) Gentle denoise
        gray = cv2.fastNlMeansDenoising(gray, h=8, templateWindowSize=7, searchWindowSize=21)

        # 4) Sharpen (unsharp mask)
        blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
        sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

        # 5) Threshold candidates
        otsu_thr, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adap = cv2.adaptiveThreshold(
            sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12
        )

        # 6) Morphology to clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        otsu_clean = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel, iterations=1)
        adap_clean = cv2.morphologyEx(adap, cv2.MORPH_OPEN, kernel, iterations=1)

        # 7) Pick the better mask by balance of foreground pixels
        def score(img_bin: np.ndarray) -> float:
            white = np.count_nonzero(img_bin == 255)
            total = img_bin.size
            ratio = white / max(1, total)
            # Prefer ~60% white coverage
            return -abs(ratio - 0.6)

        candidate = adap_clean if score(adap_clean) >= score(otsu_clean) else otsu_clean

        processed_path = os.path.join(out_dir, f"processed_{os.path.basename(src_path)}.jpg")
        cv2.imwrite(processed_path, candidate)
        return processed_path
    except Exception:
        # Fallback: return original if preprocessing fails
        return src_path


def _guess_mime_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip('.')
    mapping = {
        'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png', 'gif': 'image/gif',
        'bmp': 'image/bmp', 'tiff': 'image/tiff', 'tif': 'image/tiff', 'webp': 'image/webp',
        'heic': 'image/heic', 'heif': 'image/heif'
    }
    return mapping.get(ext, 'application/octet-stream')


def _normalize_expiry(exp_str: str) -> str:
    """Normalize various expiry formats to MM/YY (or MM/YYYY if 4-digit year)."""
    if not exp_str:
        return ''
    s = exp_str.strip().lower()
    # remove labels
    for lab in ['exp', 'expiry', 'exp.', 'best before', 'use by', 'bb', ':']:
        s = s.replace(lab, ' ')
    s = re.sub(r'\s+', ' ', s).strip()

    # Month name mapping
    months = {
        'jan': 1, 'january': 1,
        'feb': 2, 'february': 2,
        'mar': 3, 'march': 3,
        'apr': 4, 'april': 4,
        'may': 5,
        'jun': 6, 'june': 6,
        'jul': 7, 'july': 7,
        'aug': 8, 'august': 8,
        'sep': 9, 'sept': 9, 'september': 9,
        'oct': 10, 'october': 10,
        'nov': 11, 'november': 11,
        'dec': 12, 'december': 12,
    }

    # 1) Try numeric forms: MM/YY, MM-YY, MM/YYYY, DD/MM/YY etc.
    m = re.search(r'(\b\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})\b', s)
    if m:
        # Prefer last two as month, year if looks like DD/MM/YY; swap if needed
        a, b, c = m.groups()
        day, mon, year = a, b, c
        # Heuristic: if first > 12, then it's day
        if int(day) > 12 and int(mon) <= 12:
            pass
        else:
            # Could be MM/DD/YY; prefer first as month when plausible
            mon, day = day, mon
        mon_i = max(1, min(12, int(mon)))
        y = int(year)
        if y < 100:
            y = 2000 + y
        return f"{mon_i:02d}/{str(y)[-2:]}"

    m = re.search(r'(\b\d{1,2})[\/\-](\d{2,4})\b', s)
    if m:
        mon, year = m.groups()
        mon_i = max(1, min(12, int(mon)))
        y = int(year)
        if y < 100:
            y = 2000 + y
        return f"{mon_i:02d}/{str(y)[-2:]}"

    # 2) Month name + year, e.g., Aug 2025 or September-25
    m = re.search(r'\b([a-zA-Z]{3,9})\s*[-\/]?\s*(\d{2,4})\b', s)
    if m:
        mon_name, year = m.groups()
        mon_i = months.get(mon_name.lower(), 0)
        if mon_i:
            y = int(year)
            if y < 100:
                y = 2000 + y
            return f"{mon_i:02d}/{str(y)[-2:]}"

    return ''


def _expiry_to_human(exp_str: str) -> str:
    """Convert normalized MM/YY or MM/YYYY to 'Mon YYYY' (e.g., 11/25 -> Nov 2025)."""
    if not exp_str:
        return ''
    try:
        parts = exp_str.split('/')
        if len(parts) != 2:
            return ''
        mon = max(1, min(12, int(parts[0])))
        yy = int(parts[1])
        year = 2000 + yy if yy < 100 else yy
        month_names = [
            'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'
        ]
        return f"{month_names[mon-1]} {year}"
    except Exception:
        return ''


def gemini_ocr_extract(image_path: str) -> dict:
    """
    Use Gemini multimodal to extract plain text from the label and parse key fields.
    Returns dict: { 'raw_text': str, 'name': str|None, 'expiry': str|None }
    """
    try:
        api_key = os.getenv('GEMINI_API_KEY', 'AIzaSyDUG9Gb_54OKrZmZOGr3ovxX-Rqpa3Tpn8')
        if not api_key:
            return { 'raw_text': '', 'name': None, 'expiry': None }
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel('gemini-1.5-flash')
        mime = _guess_mime_type(image_path)
        with open(image_path, 'rb') as f:
            data = f.read()

        # Ask Gemini to return structured JSON to avoid heuristic name picking
        prompt = (
            "You are extracting information from a medicine label image. "
            "Return STRICT JSON with keys: medicine_name (string), expiry (string or null), raw_text (string). "
            "Medicine name should be the brand/generic name printed as the product name, not price or taxes. "
            "If expiry is present (MM/YY, MM-YY, or MM/YYYY), include it; otherwise null. "
            "Do not include any extra text outside JSON."
        )

        response = model.generate_content([
            { 'text': prompt },
            { 'inline_data': { 'mime_type': mime, 'data': data } }
        ])

        raw_response = (response.text or '').strip()
        name = None
        expiry = None
        raw = ''
        import json as _json
        try:
            data = _json.loads(raw_response)
            name = (data.get('medicine_name') or '').strip() or None
            expiry = (data.get('expiry') or '').strip() or None
            raw = (data.get('raw_text') or '').strip()
        except Exception:
            # Fallback to plain text if model didn't comply; return text only
            raw = raw_response

        # Parse expiry from raw text using broader patterns and normalize
        expiry = None
        # Common label lines to scan first
        candidate_lines = []
        for line in (raw.splitlines() if raw else []):
            if re.search(r'\b(exp|expiry|exp\.|best before|use by)\b', line, flags=re.I):
                candidate_lines.append(line)
        candidate_text = '\n'.join(candidate_lines) if candidate_lines else raw

        # Try normalization from candidate text
        m_all = re.findall(r'(?:exp(?:iry)?\.?\s*[:\-]?\s*)?([a-zA-Z]{3,9}\s*[-\/]?\s*\d{2,4}|\d{1,2}[\/\-]\d{2,4}|\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})', candidate_text or '', flags=re.I)
        for token in m_all or []:
            norm = _normalize_expiry(token)
            if norm:
                expiry = norm
                break

        return { 'raw_text': raw, 'name': name, 'expiry': expiry }
    except Exception:
        return { 'raw_text': '', 'name': None, 'expiry': None }


def gemini_expiry_only(raw_text: str) -> str:
    """Ask Gemini to return only expiry in MM/YY from provided raw OCR text."""
    try:
        api_key = os.getenv('GEMINI_API_KEY', 'AIzaSyDUG9Gb_54OKrZmZOGr3ovxX-Rqpa3Tpn8')
        if not api_key:
            return ''
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = (
            "From the following medicine label text, extract the EXPIRY date only and return in MM/YY. "
            "If year is four digits, convert to last two digits. If not confidently present, return empty string.\n\n"
            f"TEXT:\n{raw_text}"
        )
        resp = model.generate_content(prompt)
        cand = (resp.text or '').strip()
        norm = _normalize_expiry(cand)
        return norm
    except Exception:
        return ''


def gemini_name_only(image_path: str) -> str:
    """Ask Gemini to return only the medicine product name as plain text as a last-resort fallback."""
    try:
        api_key = os.getenv('GEMINI_API_KEY', 'AIzaSyDUG9Gb_54OKrZmZOGr3ovxX-Rqpa3Tpn8')
        if not api_key:
            return ''
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        mime = _guess_mime_type(image_path)
        with open(image_path, 'rb') as f:
            data = f.read()
        prompt = (
            "From this medicine label image, output ONLY the product's medicine name (brand or generic). "
            "Do not include price, quantity, taxes, dosage strength, or any other words. Return only the name."
        )
        response = model.generate_content([
            { 'text': prompt },
            { 'inline_data': { 'mime_type': mime, 'data': data } }
        ])
        return (response.text or '').strip()
    except Exception:
        return ''

## Removed EasyOCR-based extraction; Gemini Vision will be used exclusively

def get_medicine_info_with_gemini(medicine_name):
    """
    Uses Google Gemini AI to get comprehensive medicine information.
    """
    try:
        # Configure Gemini API
        api_key = os.getenv('GEMINI_API_KEY', 'AIzaSyDUG9Gb_54OKrZmZOGr3ovxX-Rqpa3Tpn8')
        
        if not api_key:
            return "API key not configured. Please set up your Gemini API key."
        
        genai.configure(api_key=api_key)
        
        # Initialize the model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Create a concise prompt for medicine information
        prompt = f"""
        Provide brief, essential information about {medicine_name} in this exact format:
        
        **Purpose:** [1-2 sentences on what it treats]
        **How it works:** [1 sentence on mechanism]
        **Common side effects:** [Top 3-4 side effects]
        **Key warnings:** [Most important 2-3 warnings]
        **Typical dose:** [Standard adult dose]
        **Interactions:** [Most important 2-3 drug interactions]
        
        Keep each section to 1-2 sentences maximum. Be concise and practical.
        """
        
        # Generate response
        response = model.generate_content(prompt)
        
        return response.text
        
    except Exception as e:
        return f"Error getting information from Gemini AI: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess for WhatsApp-like images
        processed_path = preprocess_image(filepath, app.config['UPLOAD_FOLDER'])
        
        try:
            engine_used = 'gemini'
            # Use Gemini OCR for text extraction
            g = gemini_ocr_extract(processed_path)
            gemini_text = g.get('raw_text') or None
            name = g.get('name')
            expiry = g.get('expiry')
            if not expiry and g.get('raw_text'):
                expiry = gemini_expiry_only(g.get('raw_text') or '') or None
            # Convert to human-readable format
            if expiry:
                human = _expiry_to_human(expiry)
                expiry = human or expiry

            # Build simple OCR list for UI from Gemini raw text (line-based)
            ocr_results = []
            if gemini_text:
                for line in [l for l in gemini_text.splitlines() if l.strip()]:
                    # emulate structure: (bbox, text, conf) -> but we only need text/conf for UI
                    ocr_results.append(([[0,0],[0,0],[0,0],[0,0]], line.strip(), 0.99))
            
            # Last-resort fallback: ask Gemini for name only if missing
            if not name:
                inferred_name = gemini_name_only(processed_path)
                if inferred_name:
                    name = inferred_name

            if name:
                # Get AI-generated medicine information
                medicine_info = get_medicine_info_with_gemini(name)
                
                # Convert image to base64 for display
                with open(filepath, 'rb') as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                
                # Clean up uploaded/processed files
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    if processed_path != filepath and os.path.exists(processed_path):
                        os.remove(processed_path)
                except Exception:
                    pass
                
                # Convert OCR results (from Gemini lines) to JSON-serializable types
                safe_ocr_results = []
                for bbox, text, conf in ocr_results:
                    safe_bbox = [[int(0), int(0)] for _ in range(4)]
                    safe_conf = float(conf)
                    safe_ocr_results.append({
                        'text': str(text),
                        'confidence': round(safe_conf, 2),
                        'bbox': safe_bbox,
                    })

                return jsonify({
                    'success': True,
                    'medicine_name': str(name),
                    'expiry_date': str(expiry) if expiry else 'Not detected',
                    'medicine_info': medicine_info,
                    'ocr_results': safe_ocr_results,
                    'engine_used': engine_used,
                    'gemini_text': gemini_text,
                    'image': f"data:image/jpeg;base64,{img_base64}"
                })
            else:
                # Graceful success with placeholders instead of error to avoid breaking UI
                with open(filepath, 'rb') as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

                # Clean up uploaded/processed files
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    if processed_path != filepath and os.path.exists(processed_path):
                        os.remove(processed_path)
                except Exception:
                    pass

                safe_ocr_results = []
                if gemini_text:
                    for line in [l for l in gemini_text.splitlines() if l.strip()]:
                        safe_ocr_results.append({
                            'text': line.strip(),
                            'confidence': 0.99,
                            'bbox': [[0,0],[0,0],[0,0],[0,0]]
                        })

                return jsonify({
                    'success': True,
                    'medicine_name': 'Not detected',
                    'expiry_date': 'Not detected',
                    'medicine_info': 'Could not determine medicine name from the image.',
                    'ocr_results': safe_ocr_results,
                    'engine_used': engine_used,
                    'gemini_text': gemini_text,
                    'image': f"data:image/jpeg;base64,{img_base64}"
                })
                
        except Exception as e:
            # Clean up uploaded/processed files
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                if processed_path != filepath and os.path.exists(processed_path):
                    os.remove(processed_path)
            except Exception:
                pass
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type. Please upload a valid image file.'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)
