import pandas as pd
from pymongo import MongoClient
import calendar
import datetime
import re
import os
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

UPLOAD_FOLDER = 'uploads/x_ray_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm', 'dicom'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

x_ray_bp = Blueprint('x_ray_analyze', __name__, url_prefix='/api')


def get_unique_filename(filename):
    """
    This function will check if the file already exists in the upload folder.
    If it does, it appends a serial number to the filename to make it unique.
    """
    basename, extension = os.path.splitext(filename)
    new_filename = filename
    count = 1

    # Check if file already exists
    while os.path.exists(os.path.join(UPLOAD_FOLDER, new_filename)):
        new_filename = f"{basename}_{count}{extension}"
        count += 1

    return new_filename


@x_ray_bp.route('/x_ray_analyze', methods=['POST'])
def analyze_xray():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Get a unique filename if the file already exists
        unique_filename = get_unique_filename(filename)
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(file_path)

        # Assume you have some image analysis function here
        analysis_result = perform_analysis(file_path)

        # You can return the path of the uploaded image for preview
        return jsonify({'imageUrl': f'/uploads/{filename}', 'analysis': analysis_result})

    return jsonify({'error': 'File not allowed'}), 400

def perform_analysis(file_path):
    # Perform your X-ray analysis logic here (machine learning, etc.)
    # For now, we'll return a mock result
    return {"result": "positive", "severity": "moderate"}



