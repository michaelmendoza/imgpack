# app.py
import os
from typing import Optional, List

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import HTMLResponse, FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from imgpack.utils import encode_data, pack_envelope

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8765"))

app = FastAPI(title="Imgpack demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# --- Simple storage for "current array" (for demo only) ---
_current_array: Optional[np.ndarray] = None

def default_array(h: int = 256, w: int = 384) -> np.ndarray:
    """
    Nice looking gradient/sine pattern for testing.
    """
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    img = 0.5 + 0.5*np.sin(10*(x**2 + y**2)) * np.exp(-2*((x-0.5)**2 + (y-0.5)**2))
    return img

@app.get("/", response_class=HTMLResponse)
def index_html():
    # Serve the static HTML file
    return FileResponse("./demo/index.html")

@app.post("/set_array")
def set_array(
    data: List[List[float]] = Body(..., embed=True, description="2D list of numbers"),
):
    """
    Provide a 2D array (list-of-lists) to use for the next WebSocket send.
    Example:
      POST /set_array
      {"data": [[0,1,2],[3,4,5]]}
    """
    global _current_array
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        return PlainTextResponse("data must be a 2D list", status_code=400)
    _current_array = arr
    return {"ok": True, "shape": list(arr.shape)}

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        # Pick array: either user-provided or a default generated one
        arr = _current_array if _current_array is not None else default_array()

        # Quantize → envelope
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))
        blob, header = encode_data(arr, vmin=vmin, vmax=vmax, dtype="uint16")
        envelope = pack_envelope(blob, header)

        # Send once on connect (you can extend this to stream periodically)
        await ws.send_bytes(envelope)

        # Keep the socket open; echo "send" to resend (handy for manual testing)
        while True:
            _ = await ws.receive_text()
            await ws.send_bytes(envelope)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        # Optionally send an error text (client ignores non-binary)
        try:
            await ws.send_text(f"error: {e}")
        except Exception:
            pass
    finally:
        await ws.close()
