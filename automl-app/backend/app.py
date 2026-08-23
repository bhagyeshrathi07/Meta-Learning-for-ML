import os
import json
import uuid
import glob
import hmac
import threading
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, send_file, Response, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import custom modules
from pipeline import run_automl
from codegen import generate_training_script

# Check if running in Docker (static folder exists)
STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
IS_DOCKER = os.path.exists(STATIC_FOLDER)

if IS_DOCKER:
    app = Flask(__name__, static_folder='static', static_url_path='')
else:
    app = Flask(__name__)

CORS(app)

# --- API KEY CONFIGURATION ---
API_KEY = os.getenv('AUTOML_API_KEY')
if not API_KEY:
    print("⚠️  WARNING: AUTOML_API_KEY not set in .env file. API is unprotected!")


def require_api_key(f):
    """Decorator to protect endpoints with API key validation."""
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')

        if not API_KEY:
            # If no API key configured, allow all requests (development mode)
            return f(*args, **kwargs)

        if not api_key:
            return jsonify({"error": "API key required", "code": "MISSING_API_KEY"}), 401

        # Use timing-safe comparison to prevent timing attacks
        if not hmac.compare_digest(api_key, API_KEY):
            return jsonify({"error": "Invalid API key", "code": "INVALID_API_KEY"}), 401

        return f(*args, **kwargs)
    return decorated

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Increase Upload Limit for Big CSVs (100MB)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 

# Thread Executor (Prevents blocking main thread)
executor = ThreadPoolExecutor(max_workers=2)
# In-memory storage for running tasks
tasks = {}

# --- CLEANUP ON STARTUP ---
# Delete old files so the server starts fresh
for f in glob.glob(os.path.join(MODELS_FOLDER, "*.pkl")):
    try: os.remove(f)
    except: pass
for f in glob.glob(os.path.join(UPLOAD_FOLDER, "*.csv")):
    try: os.remove(f)
    except: pass
print("🧹 Server Cleaned: Old temp files removed.")

# --- BACKGROUND WORKER ---
def background_task(task_id, filepath, target, selected_models):
    """Runs the pipeline in a separate thread and handles cleanup."""
    try:
        def update_progress(progress, message):
            # Callback to update global task state
            if task_id in tasks:
                tasks[task_id]['progress'] = progress
                tasks[task_id]['logs'].append(message)

        # Run the heavy ML logic
        results = run_automl(filepath, target, selected_models, callback=update_progress)
        
        # Success State
        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['progress'] = 100
        tasks[task_id]['results'] = results
        tasks[task_id]['logs'].append("✅ Pipeline completed successfully.")

    except Exception as e:
        # Failure State
        print(f"Task Failed: {e}")
        tasks[task_id]['status'] = 'failed'
        tasks[task_id]['error'] = str(e)
        tasks[task_id]['logs'].append(f"❌ Error: {str(e)}")

    finally:
        # AUTO-DESTRUCT: Remove the user's CSV for privacy & space
        if os.path.exists(filepath):
            try: os.remove(filepath)
            except: pass
            print(f"🗑️ Deleted temp file: {filepath}")

# --- ROUTES ---
@app.route('/validate-key', methods=['POST'])
def validate_key():
    """Endpoint for frontend to validate API key before allowing access."""
    api_key = request.headers.get('X-API-Key')

    if not API_KEY:
        # No API key configured - always valid (development mode)
        return jsonify({"valid": True, "message": "API key validation disabled"})

    if not api_key:
        return jsonify({"valid": False, "message": "API key required"}), 401

    if hmac.compare_digest(api_key, API_KEY):
        return jsonify({"valid": True, "message": "API key validated successfully"})
    else:
        return jsonify({"valid": False, "message": "Invalid API key"}), 401


@app.route('/upload', methods=['POST'])
@require_api_key
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    target = request.form.get('target')
    models_json = request.form.get('models')
    
    try:
        selected_models = json.loads(models_json) if models_json else None
    except:
        selected_models = None

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save file temporarily with unique ID
    filename = f"{uuid.uuid4()}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Initialize Task
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "processing",
        "progress": 0,
        "logs": ["🚀 Upload received. Initializing pipeline..."],
        "results": None
    }

    # Offload to background thread
    executor.submit(background_task, task_id, filepath, target, selected_models)

    return jsonify({"task_id": task_id, "message": "Processing started"})

@app.route('/status/<task_id>', methods=['GET'])
@require_api_key
def get_status(task_id):
    """Frontend polls this endpoint to get progress bar updates"""
    task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(task)

@app.route('/download', methods=['GET'])
@require_api_key
def download_model():
    """Downloads the .pkl file"""
    model_name = request.args.get('model')
    if not model_name:
        return jsonify({"error": "Model name required"}), 400

    filename = model_name.replace(" ", "_") + ".pkl"
    
    # Absolute Path to prevent 'File Not Found'
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, MODELS_FOLDER, filename)
    
    if not os.path.exists(path):
        return jsonify({"error": f"Model file not found: {filename}"}), 404

    return send_file(path, as_attachment=True)

@app.route('/download-code', methods=['GET'])
@require_api_key
def download_code():
    """Generates a Python script to reproduce the model"""
    model_name = request.args.get('model')
    task_id = request.args.get('task_id')

    if not model_name or not task_id:
        return jsonify({"error": "Missing parameters"}), 400

    task = tasks.get(task_id)
    if not task or not task.get('results'):
        return jsonify({"error": "Task results not found"}), 404

    # Find the model configuration in the results
    model_data = next((m for m in task['results'] if m["Model"] == model_name), None)
    
    if not model_data:
        return jsonify({"error": "Model data not found"}), 404

    # Generate the code string
    script_content = generate_training_script(
        model_name,
        model_data.get("Best Params", {}),
        model_data.get("Task Type", "Classification"),
        "target_column" # Placeholder name
    )

    return Response(
        script_content,
        mimetype="text/x-python",
        headers={"Content-disposition": f"attachment; filename=train_{model_name.replace(' ', '_')}.py"}
    )

# --- SERVE REACT APP (Docker/Production) ---
if IS_DOCKER:
    @app.route('/')
    def serve_react():
        return send_from_directory(app.static_folder, 'index.html')

    @app.errorhandler(404)
    def not_found(e):
        # Serve React app for client-side routing
        return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    # In Docker, bind to 0.0.0.0 to accept external connections
    host = '0.0.0.0' if IS_DOCKER else '127.0.0.1'
    app.run(debug=not IS_DOCKER, host=host, port=5000)