
import sys
import os

# Add backend directory to path so imports work
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify
from flask_cors import CORS
from question_generator import generate_quiz

app = Flask(__name__)

# Enable CORS so the frontend HTML file can call this API
CORS(app)


@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "running", "message": "Quiz Generator API is up!"})


@app.route('/generate-quiz', methods=['POST'])
def generate_quiz_endpoint():
    """
    Main API endpoint for quiz generation.

    Accepts JSON body:
      {
        "text": "Your academic text here...",
        "num_mcq": 5,       (optional, default 5)
        "num_tf": 5,        (optional, default 5)
        "num_fill": 5       (optional, default 5)
      }

    Returns JSON:
      {
        "mcq": [...],
        "true_false": [...],
        "fill_blanks": [...]
      }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON body received."}), 400

        text = data.get('text', '').strip()

        if not text:
            return jsonify({"error": "The 'text' field is required and cannot be empty."}), 400

        # Optional parameters with safe defaults
        num_mcq  = int(data.get('num_mcq', 5))
        num_tf   = int(data.get('num_tf', 5))
        num_fill = int(data.get('num_fill', 5))

        # Clamp values to reasonable range
        num_mcq  = max(1, min(num_mcq, 10))
        num_tf   = max(1, min(num_tf, 10))
        num_fill = max(1, min(num_fill, 10))

        # ── Generate the quiz ──
        result = generate_quiz(text, num_mcq=num_mcq, num_tf=num_tf, num_fill=num_fill)

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


if __name__ == '__main__':
    print("=" * 50)
    print("  Automated Quiz Generator - Backend Server")
    print("  Running at: http://127.0.0.1:5000")
    print("  Health check: http://127.0.0.1:5000/health")
    print("=" * 50)
    app.run(debug=True, host='127.0.0.1', port=5000)
