import pandas as pd
from pymongo import MongoClient
import calendar
import datetime
import re
import os
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'csv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

parser_bp = Blueprint('parser_bp', __name__)


def format_month_year(mm_yy):
    if isinstance(mm_yy, (pd.Timestamp, datetime.datetime, datetime.date)):
        return f"{calendar.month_name[mm_yy.month]}-{mm_yy.year}"
    if isinstance(mm_yy, str) and re.match(r'\d{2}/\d{2}', mm_yy.strip()):
        try:
            month, year_suffix = mm_yy.split('/')
            month = int(month)
            year = 2000 + int(year_suffix)
            return f"{calendar.month_name[month]}-{year}"
        except Exception as e:
            print(f"Error parsing {mm_yy}: {e}")
            return str(mm_yy)
    return str(mm_yy)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def parse_file(file_path):
    try:
        df = pd.read_excel(file_path, engine='openpyxl', header=[2, 3])

        df.columns = [' '.join([str(a).strip() for a in col if str(a) != 'nan']) for col in df.columns]
        df = df.dropna(how='all')

        df = df.rename(columns={
            'Taluka Unnamed: 1_level_1': 'Taluka',
            'Name of SAM Child Unnamed: 3_level_1': 'Name of SAM Child',
            'PHC Unnamed: 2_level_1': 'PHC',
            'Sex Unnamed: 10_level_1': 'Sex',
            'Month and Year of Birth Month': 'Birth Month',
            'Month and Year of Birth Year': 'Birth Year',
            'If Any conginital anomaly (Name of anomaly) Name of Anomaly': 'Name of Anomaly',
            'Address of SAM Child Village': 'Village',
            'Age in months when detected as SAM Unnamed: 9_level_1': 'Age in months when detected as SAM',
            'If any serious disease Name of Disease': 'Name of Disease',
            'admited at NRC or not (Yes/NO) Unnamed: 18_level_1': 'admited at NRC or not (Yes/NO)'
        })

        for col in ['Name of Anomaly', 'Name of Disease', 'Village']:
            df[col] = df[col].astype(str).str.strip().replace({'None': 'No', 'nan': 'No'})

        df = df.dropna(subset=[
            'Name of SAM Child', 'PHC', 'Taluka', 'Sex', 'Birth Month', 'Birth Year',
            'Name of Anomaly', 'Village', 'Age in months when detected as SAM',
            'Name of Disease', 'admited at NRC or not (Yes/NO)'
        ])

        df['Birth Month'] = pd.to_numeric(df['Birth Month'], errors='coerce').astype('Int64')
        df['Birth Year'] = pd.to_numeric(df['Birth Year'], errors='coerce').astype('Int64')
        df = df.dropna(subset=['Birth Month', 'Birth Year'])

        df_final = df[[
            'Name of SAM Child', 'PHC', 'Taluka', 'Sex',
            'Birth Month', 'Birth Year', 'Name of Anomaly', 'Village',
            'Age in months when detected as SAM', 'Name of Disease',
            'admited at NRC or not (Yes/NO)'
        ]].rename(columns={'Birth Month': 'Month', 'Birth Year': 'Year'})

        combined_records = []

        for idx, row in df_final.iterrows():
            base_record = row.to_dict()
            original_row = df.loc[idx]

            details = {}
            for col, val in original_row.items():
                if any(skip_key in col.lower() for skip_key in ['name', 'address', 'age', 'month']):
                    continue

                parts = col.split(" ")
                if len(parts) < 2:
                    continue

                month_raw = parts[0]
                field_name = ' '.join(parts[1:]).strip().lower()
                formatted_month = format_month_year(month_raw.strip())

                if formatted_month not in details:
                    details[formatted_month] = {}

                if 'height' in field_name:
                    details[formatted_month]['Height'] = val
                elif 'weight' in field_name:
                    details[formatted_month]['Weight'] = val
                elif 'sam' in field_name or 'status' in field_name:
                    details[formatted_month]['Status'] = val

            clean_details = {k: v for k, v in details.items() if any(pd.notnull(val) for val in v.values())}
            base_record["Details"] = clean_details
            base_record["timestamp"] = datetime.datetime.now()

            combined_records.append(base_record)

        client = MongoClient("mongodb://localhost:27017")
        db = client["sin_waste"]
        collection = db["excel_upload"]

        if combined_records:
            # Insert and get inserted data with ObjectIDs removed
            insert_result = collection.insert_many(combined_records)
            for rec in combined_records:
                rec['_id'] = str(insert_result.inserted_ids[combined_records.index(rec)])
            return {
                "message": f"✅ Inserted {len(combined_records)} records.",
                "data": combined_records
            }
        else:
            return {"message": "⚠️ No valid records to insert.", "data": []}

    except Exception as e:
        print("❌ Error during parsing:", e)
        return {"error": str(e)}


@parser_bp.route('/upload-file', methods=['POST'])
def upload_file():
    
    print("📥 File upload request received.")
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            result = parse_file(file_path)
            return jsonify(result), 200 if "error" not in result else 500
        except Exception as e:
            print("❌ Error saving or processing file:", e)
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'File type not allowed'}), 400
