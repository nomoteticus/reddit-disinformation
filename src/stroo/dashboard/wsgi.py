"""WSGI entry point: ``gunicorn stroo.dashboard.wsgi:server``.

The data folder is taken from ``STROO_DATA_DIR`` (see ``stroo.settings``).
"""

from .app import create_app

server = create_app().server
