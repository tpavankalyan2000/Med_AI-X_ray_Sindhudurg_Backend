from flask import Flask
from flask_cors import CORS
from auth.routes import auth_bp
# from Parsers.parser import parser_bp
from X_Ray_Routes.classification import x_ray_bp
from X_Ray_Routes.converter_x_ray_images import convert_bp
from X_Ray_Routes.x_ray_analyze import x_ray_data_bp
from X_Ray_Routes.stats_dashboard import stats_dash_bp

app = Flask(__name__)
CORS(app)  # <--- Enable CORS for all routes

app.register_blueprint(auth_bp, url_prefix='/api')
# app.register_blueprint(parser_bp)
app.register_blueprint(x_ray_bp, url_prefix='/api')
app.register_blueprint(convert_bp, url_prefix='/api')
app.register_blueprint(x_ray_data_bp, url_prefix='/api')
app.register_blueprint(stats_dash_bp, url_prefix='/api')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5555)

