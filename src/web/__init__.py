# src/web/__init__.py

from flask import Blueprint

web_bp = Blueprint('web', __name__)

from . import routes, export_routes  # Import routes to register them with the blueprint

# Register the export routes
web_bp.register_blueprint(export_routes.export_routes)