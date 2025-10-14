# src/web/__init__.py

from flask import Blueprint

web_bp = Blueprint('web', __name__)

from . import routes  # Import routes to register them with the blueprint