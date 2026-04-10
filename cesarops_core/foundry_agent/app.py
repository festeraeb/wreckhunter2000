from flask import Flask, request, jsonify
import os
import requests

app = Flask(__name__)

MODEL_ENDPOINT = os.environ.get("MODEL_ENDPOINT", "")
MODEL_KEY = os.environ.get("MODEL_KEY", "")


@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    prompt = data.get("input", "")

    # If no model configured, return echo-mode to allow quick testing
    if not MODEL_ENDPOINT or not MODEL_KEY:
        return jsonify({"reply": f"[echo-mode] {prompt}"})

    headers = {"Authorization": f"Bearer {MODEL_KEY}", "Content-Type": "application/json"}
    payload = {"input": prompt}
    try:
        resp = requests.post(MODEL_ENDPOINT, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
