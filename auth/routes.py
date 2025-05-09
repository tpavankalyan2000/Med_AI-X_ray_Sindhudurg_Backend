from flask import Blueprint, request, jsonify
from database import MongoDB
from werkzeug.security import generate_password_hash
from werkzeug.security import check_password_hash  # This is for comparing hashed passwords


auth_bp = Blueprint('auth', __name__, url_prefix='/api')
mongo = MongoDB()


@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    # Check if email and password are provided
    if not email or not password:
        return jsonify(message="Email and password are required"), 400

    # Connect to the users collection
    users_collection = mongo.db.users

    # Find user by email
    user = users_collection.find_one({"email": email})
    
    if not user:
        return jsonify(message="Invalid email or password"), 401

    # Check if the password matches
    if not check_password_hash(user['password'], password):
        return jsonify(message="Invalid email or password"), 401

    # Successful login
    return jsonify(message="Login successful", user={"email": user['email'], "role": user.get('role', 'user')}), 200

@auth_bp.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()  # Get data from frontend

    # Check if all necessary fields are present
    required_fields = [
        "name", "email", "mobile", "telephone", "state", "district", 
        "taluka", "village", "role", "address", "password", "confirmPassword"
    ]
    
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"{field} is required"}), 400
    
    # Validate password confirmation
    if data['password'] != data['confirmPassword']:
        return jsonify({"error": "Passwords do not match"}), 400

    # Hash the password before saving
    hashed_password = generate_password_hash(data['password'])

    # Prepare the data for MongoDB insertion
    user_data = {
        "name": data["name"],
        "email": data["email"],
        "mobile": data["mobile"],
        "telephone": data["telephone"],
        "state": data["state"],
        "district": data["district"],
        "taluka": data["taluka"],
        "village": data["village"],
        "role": data["role"],
        "address": data["address"],
        "password": hashed_password,
    }

    # Insert the user into the MongoDB collection
    try:
        mongo.db.users.insert_one(user_data)
        return jsonify({"message": "User successfully created"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500




