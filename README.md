# ImgPack

A tiny binary format for packing and streaming 2D NumPy arrays (images) from Python to the browser.  
Built with **FastAPI + WebSockets** on the server and **TypedArrays + Canvas** in the browser. 

---

## 🚀 Quick Start

### 1. Clone & install
```bash
git clone https://github.com/michaelmendoza/imgpack
cd imgpack
pip install -r requirements.txt
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
from utils_bin import encode_data, pack_envelope

# make a test image
arr = np.linspace(0, 1, 64*64, dtype=np.float32).reshape(64, 64)

# encode and pack into envelope
blob, header = encode_data(arr, vmin=0.0, vmax=1.0, dtype="uint16")
envelope = pack_envelope(blob, header)

# send `envelope` over a WebSocket:
# await ws.send_bytes(envelope)
```

### Python (client-side decode)

```python
from utils_bin import unpack_envelope, blob_to_ndarray

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

MAGIC(4) | VERSION(1) | HEADER\_LEN(4, BE) | HEADER\_JSON(UTF-8) | PAD | BLOB

````

- **MAGIC**: `"VCBN"` (4 bytes)  
- **VERSION**: format version (1 byte)  
- **HEADER_LEN**: JSON header size (uint32, big-endian)  
- **HEADER_JSON**: metadata (`dtype`, `shape`, `resolution`, etc.)  
- **PAD**: ensures BLOB starts on 2-byte boundary (for `uint16` alignment)  
- **BLOB**: raw quantized array data (little-endian)  

---