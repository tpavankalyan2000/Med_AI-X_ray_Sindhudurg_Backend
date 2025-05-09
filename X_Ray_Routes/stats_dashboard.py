from flask import Blueprint, jsonify

from database import MongoDB

stats_dash_bp = Blueprint('stats_dash', __name__)

mongo = MongoDB()

@stats_dash_bp.route('/stats_dashboard', methods=['GET'])
def stats_dash():
    xray_collection = mongo.db.x_ray_analyses
    query = {}
    docs = xray_collection.find(query)
    counts_of_doc = 0
    for doc in docs:
        print(doc)
        counts_of_doc += 1
    print(counts_of_doc)
    results = {
                # "chat_sessions": 0,
                "xray_analyses": counts_of_doc, 
                # "extracted_images": 0, 
                "documents_analysed": 0}
    return jsonify({"results": results})