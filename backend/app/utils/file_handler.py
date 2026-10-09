import os
import uuid
from werkzeug.utils import secure_filename
from flask import current_app

def allowed_file(filename, allowed_extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions

def save_uploaded_file(file, subfolder="images"):
    """
    Saves uploaded file with unique name and returns path + original name.
    """
    if not file or file.filename == "":
        return None, None

    original_name = secure_filename(file.filename)
    ext = original_name.rsplit(".", 1)[1].lower() if "." in original_name else "bin"
    unique_name = f"{uuid.uuid4().hex}.{ext}"

    folder = os.path.join(current_app.config["UPLOAD_FOLDER"], subfolder)
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, unique_name)
    file.save(filepath)
    return filepath, original_name

def get_file_url(filepath):
    """Return relative path usable by frontend."""
    if not filepath:
        return None
    return filepath.replace("\\", "/")
