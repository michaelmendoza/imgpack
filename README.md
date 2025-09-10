# ImgPack

A tiny binary format for packing and streaming 2D NumPy arrays (images) from Python to the browser.  
Built with **FastAPI + WebSockets** on the server and **TypedArrays + Canvas** in the browser.  

Currently, ImgPack supports **lossy quantization** into either **8-bit** (`uint8`, 0–255) or **16-bit** (`uint16`, 0–4096) integer buffers.  
Your original float/complex arrays are scaled into this range based on `vmin`/`vmax` before being sent.  
This keeps payloads compact and easy to render in browsers, but it means the transport is **not lossless**.

---

## Install

```bash
git clone https://github.com/michaelmendoza/imgpack
````

## 🚀 Quick Start with Demo

### 1. Clone & install demo dependencies
```bash
pip install fastapi uvicorn 
git clone https://github.com/michaelmendoza/imgpack
cd imgpack
````

### 2. Run the demo server

```bash
python run.py
```

Server runs at [http://localhost:8765](http://localhost:8765).

### 3. Open the browser

Visit [http://localhost:8765](http://localhost:8765) and you’ll see a grayscale image rendered from a NumPy array streamed over a WebSocket.

---

## 🖼 Examples

### Python (server-side encode + pack)

```python
import numpy as np
from imgpack.utils import encode_data, pack_envelope

# make a test image
arr = np.linspace(0, 1, 64*64, dtype=np.float32).reshape(64, 64)

# encode and pack into envelope (quantized to 16-bit)
blob, header = encode_data(arr, vmin=0.0, vmax=1.0, dtype="uint16")
envelope = pack_envelope(blob, header)

# send `envelope` over a WebSocket:
# await ws.send_bytes(envelope)
```

### Python (client-side decode)

```python
from imgpack.utils import unpack_envelope, blob_to_ndarray

# Suppose you received `envelope` as bytes
header, blob = unpack_envelope(envelope)
arr_decoded = blob_to_ndarray(header, blob)

print("Decoded shape:", arr_decoded.shape)
print("Decoded dtype:", arr_decoded.dtype)
print("Header:", header)
```

Output:

```
Decoded shape: (64, 64)
Decoded dtype: uint16
Header: {'fmt': 'vcbin', 'version': 1, 'dtype': 'uint16', 'endianness': 'LE', ...}
```

### JavaScript (browser decode)

```js
ws.onmessage = async (evt) => {
  const ab = evt.data instanceof Blob ? await evt.data.arrayBuffer() : evt.data;
  const { header, blobBytes } = readHeaderAndBlob(ab);
  const typed = blobToTypedArray(header, blobBytes);
  renderGrayscale(canvas, typed, header.shape, header);
};
```

---

## 📦 Binary Format

```
MAGIC(4) | VERSION(1) | HEADER_LEN(4, BE) | HEADER_JSON(UTF-8) | PAD | BLOB
```

* **MAGIC**: `"VCBN"` (4 bytes)
* **VERSION**: format version (1 byte)
* **HEADER\_LEN**: JSON header size (uint32, big-endian)
* **HEADER\_JSON**: metadata (`dtype`, `shape`, `resolution`, etc.)
* **PAD**: ensures BLOB starts on a 2-byte boundary (for `uint16` alignment)
* **BLOB**: raw quantized array data (`uint8` or `uint16`, little-endian)

---

## ⚠️ Limitations

* Only supports **8-bit** and **16-bit quantized encodings** for now.
* Quantization is **lossy**: values outside `[vmin, vmax]` are clipped.
