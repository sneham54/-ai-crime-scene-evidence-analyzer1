import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(
        debug=False,
        use_reloader=False,
        threaded=True,
        host=os.getenv("FLASK_HOST", "0.0.0.0"),
        port=int(os.getenv("FLASK_PORT", "5000")),
    )
