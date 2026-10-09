"""Flask development entry point.

Run this file from the backend directory with ``python app.py``.
The application implementation lives in the sibling ``app`` package; importing
it by its package name keeps all route and service imports in one namespace.
"""

from app import create_app


app = create_app()


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False, threaded=True, host="0.0.0.0", port=5000)
