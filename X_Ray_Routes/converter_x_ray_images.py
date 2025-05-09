
from flask import Flask, request, jsonify, send_file, redirect, url_for, Blueprint
from werkzeug.utils import secure_filename
from pydicom import dcmread
from PIL import Image
import numpy as np
import os

convert_bp = Blueprint('convert_x_ray_images', __name__, url_prefix='/api')


CONVERT_UPLOAD_FOLDER = 'convert/dcm'
DOWNLOAD_IMAGE = 'convert/jpg'

os.makedirs(DOWNLOAD_IMAGE, exist_ok=True)
os.makedirs(CONVERT_UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'dcm', 'dicom'}

def allowed_file_convert(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@convert_bp.route('/convert_dcm_to_jpeg', methods=['POST'])
def convert_dcm_to_jpeg():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file_convert(file.filename):
        filename = secure_filename(file.filename)
        dcm_path = os.path.join(CONVERT_UPLOAD_FOLDER, filename)
        file.save(dcm_path)

        # Read the DICOM file
        dicom_data = dcmread(dcm_path)

        # Convert the DICOM pixel data to a NumPy array
        image = dicom_data.pixel_array

        # Check if the image is in 'I;16' format and convert to 'L' (8-bit grayscale)
        if image.dtype == np.int16:
            # Normalize the 16-bit image to fit the 8-bit range (0-255)
            image = np.uint8((image / np.max(image)) * 255)  # Normalize to [0, 255]

        # Convert the image to PIL Image with mode 'L' (8-bit grayscale)
        pil_image = Image.fromarray(image)
        pil_image = pil_image.convert('L')

        # Save the converted image as a JPEG
        jpeg_filename = filename.rsplit('.', 1)[0] + '.jpg'
        jpeg_path = os.path.join(DOWNLOAD_IMAGE, jpeg_filename)

        try:
            pil_image.save(jpeg_path, 'JPEG')
        except Exception as e:
            return jsonify({'error': f"Failed to save image: {str(e)}"}), 500

        # Return the URL of the converted image
        file_url = url_for('convert_x_ray_images.download_file', filename=jpeg_filename, _external=True)

        return jsonify({'imageUrl': file_url})
    
    return jsonify({'error': 'Invalid file format'}), 400




@convert_bp.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(DOWNLOAD_IMAGE, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

