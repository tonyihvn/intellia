from flask import Blueprint, render_template
from ..config import Config

view_routes = Blueprint('views', __name__)

@view_routes.route('/')
def index():
    return render_template('command.html')

@view_routes.route('/settings')
def settings():
    config = Config.get_llm_config()
    return render_template('settings.html', config=config)

@view_routes.route('/settings/db')
def db_config():
    config = Config.get_db_config()
    return render_template('db_config.html', config=config)

@view_routes.route('/rag')
def rag():
    return render_template('rag.html')


@view_routes.route('/db/browser')
def db_browser():
    return render_template('db_browser.html')