# /demo/app.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, List
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import HTMLResponse, FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from imgpack import encode

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / "demo"
IMGPACK_DIR = ROOT / "imgpack"

app = FastAPI(title="ImgPack Demo")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_current_array: Optional[np.ndarray] = None
_current_mode = {"dtype": "uint16", "bits": 12}  # defaults

def default_array(h=256, w=384) -> np.ndarray:
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    return 0.5 + 0.5*np.sin(10*(x**2 + y**2)) * np.exp(-2*((x-0.5)**2 + (y-0.5)**2))

@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(DEMO_DIR / "index.html")

@app.get("/decode.js")
def decode_js():
    return FileResponse(IMGPACK_DIR / "decode.js", media_type="application/javascript")

@app.post("/set_array")
def set_array(data: List[List[float]] = Body(..., embed=True)):
    global _current_array
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        return PlainTextResponse("data must be a 2D list", status_code=400)
    _current_array = arr
    return {"ok": True, "shape": list(arr.shape)}

@app.post("/set_mode")
def set_mode(dtype: str = Body(...), bits: int = Body(...)):
    _current_mode["dtype"] = dtype.lower()
    _current_mode["bits"] = int(bits)
    return {"ok": True, "mode": _current_mode}

async def handle_encode():
    arr = _current_array if _current_array is not None else default_array()
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    envelope = encode(arr, vmin=vmin, vmax=vmax,
                            dtype=_current_mode["dtype"], bits=_current_mode["bits"])
    return envelope

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            _ = await ws.receive_text()  # any message triggers resend
            envelope = await handle_encode()
            await ws.send_bytes(envelope)
            print("Sent binary data, size:", round(len(envelope)/1024/1024, ndigits=3), "MiB")

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            print("Error:", e)
            await ws.send_text(f"error: {e}")
        except Exception:
            pass
    finally:
        await ws.close()
