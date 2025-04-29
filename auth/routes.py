from flask import Blueprint, request, jsonify
import psycopg2

auth_bp = Blueprint('auth', __name__, url_prefix='/api')

PSQL_URI = "postgresql://postgres:Pavan.123%40@localhost:5432/sindrug"

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    print(email, "and" ,password)
    # Super admin check
    if email == "admin" and password == "admin":
        print("inside")
        return jsonify(message="Hello Super Admin"), 200

    # Check in PostgreSQL
    conn = psycopg2.connect(PSQL_URI)
    cursor = conn.cursor()

    query = "SELECT * FROM users WHERE email = %s AND password = %s"
    cursor.execute(query, (email, password))
    result = cursor.fetchone()

    cursor.close()
    conn.close()

    if result:
        return jsonify(message="Hello User"), 200
    else:
        return jsonify(error="Invalid Credentials"), 401


