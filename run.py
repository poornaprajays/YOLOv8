"""
YOLOv8 Object Detection Explorer
=================================
Entry point for the Flask web dashboard.

Usage:
    python run.py

Then open your browser at: http://localhost:5000
"""

from app import create_app

app = create_app()

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  🎯  YOLOv8 Object Detection Explorer")
    print("=" * 55)
    print("  Dashboard: http://localhost:5000")
    print("  Model    : YOLOv8n (nano) — auto-downloaded")
    print("  Press CTRL+C to stop the server")
    print("=" * 55 + "\n")
    # use_reloader=False fixes a watchdog crash on Python 3.13
    # (TypeError: 'handle' must be a _ThreadHandle)
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
