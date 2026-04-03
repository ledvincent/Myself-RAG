import subprocess
import sys
import time
import webbrowser

proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8000"],
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
)

time.sleep(3)  # wait for server to be ready
webbrowser.open("http://localhost:8000")

try:
    proc.wait()
except KeyboardInterrupt:
    proc.terminate()
