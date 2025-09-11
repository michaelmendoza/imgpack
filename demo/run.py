# /demo/run.py
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')
import uvicorn

if __name__ == "__main__":
    uvicorn.run("demo.app:app", host="0.0.0.0", port=8765, reload=True)
