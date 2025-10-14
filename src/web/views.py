from flask import Blueprint, render_template

view_routes = Blueprint('views', __name__)

@view_routes.route('/')
def index():
    return render_template('index.html')

@view_routes.route('/settings')
def settings():
    return render_template('settings.html')

@view_routes.route('/rag')
def rag():
    return render_template('rag.html')