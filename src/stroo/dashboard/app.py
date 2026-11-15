"""STROO? ("Is it true?"): the dashboard of posts flagged as false information.

Run locally::

    stroo dashboard            # http://127.0.0.1:8050

Serve in production (see deploy/README.md)::

    gunicorn stroo.dashboard.wsgi:server
"""

from __future__ import annotations

from pathlib import Path

from dash import Dash

from . import layout
from .callbacks import register_callbacks
from .data import load_data

DASHBOARD_VERSION = "0.2.0"


def create_app(data_dir: Path | None = None) -> Dash:
    """Build the Dash app from the six ``app_*.csv`` tables in ``data_dir``."""
    data = load_data(data_dir)
    # The tab contents are created on demand, so their callbacks refer to
    # components that are not on the page at start-up.
    app = Dash(__name__, suppress_callback_exceptions=True, title="STROO? Is it true?")
    app.layout = layout.page(data, app.get_asset_url("stroo_v1.png"), DASHBOARD_VERSION)
    register_callbacks(app, data)
    return app


def main(host: str = "127.0.0.1", port: int = 8050, debug: bool = False) -> None:
    create_app().run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main(debug=True)
