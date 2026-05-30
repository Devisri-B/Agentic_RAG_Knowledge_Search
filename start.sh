#!/usr/bin/env bash
set -e

# Start the FastAPI backend (internal, not exposed publicly)
uvicorn src.main:app --host 0.0.0.0 --port 8000 &

# Wait until the backend is ready before launching the UI,
# so the first query never races against model loading.
echo "Waiting for FastAPI backend on :8000 ..."
python - <<'PY'
import time, urllib.request
for _ in range(120):
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/", timeout=2)
        print("Backend is up.")
        break
    except Exception:
        time.sleep(2)
else:
    print("Backend did not become ready in time; starting UI anyway.")
PY

# Start the Gradio UI on the public HF Spaces port
exec python app.py
