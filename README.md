# ImgPack

A simple binary format for packing and streaming 2D NumPy arrays (images) from Python to the browser. Built with python with zero dependencies.

ImgPack supports multiple transport modes:

* **Packed**: true **n-bit bitstream** (e.g. 12-bit packed into 1.5 bytes per pixel).
* **Integer containers**: quantized to `bits` and stored in `uint8`, `uint16`, or `uint32`.
* **Float containers**: raw `float16`, `float32`, or `float64` with no quantization.

---

## Install

```bash
git clone https://github.com/michaelmendoza/imgpack
```

---

## 🚀 Quick Start with Demo

### 1. Install dependencies

```bash
pip install fastapi uvicorn numpy
```

### 2. Run the demo server

```bash
python demo/run.py
```

Server runs at [http://localhost:8765](http://localhost:8765).

### 3. Open the browser

Visit [http://localhost:8765](http://localhost:8765) and you’ll see a grayscale image rendered from a NumPy array streamed over a WebSocket.

---

## 🖼 Examples

### Python (encode + pack)

```python
import numpy as np
from imgpack import encode

arr = np.linspace(0, 1, 64*64, dtype=np.float32).reshape(64, 64)

# Example 1: true 12-bit packed stream
envelop = encode(arr, vmin=0.0, vmax=1.0, dtype="packed", bits=12)

# Example 2: 12-bit quantization into uint16 container
envelope = encode(arr, vmin=0.0, vmax=1.0, dtype="uint16", bits=12)

# Example 3: raw float32 transport (no quantization)
envelope = encode(arr, vmin=0.0, vmax=1.0, dtype="float32", bits=0)

# send `envelope` over a WebSocket:
# await ws.send_bytes(envelope)
```

### Python (decode)

```python
from imgpack import decode

arr_decoded = decode(envelope)

print("Decoded shape:", arr_decoded.shape)
print("Decoded dtype:", arr_decoded.dtype)
print("Header:", header)
```

Output (example):

```
Decoded shape: (64, 64)
Decoded dtype: uint16
Header: {
  'format': 'imgpack',
  'version': 1,
  'dtype': 'uint16',
  'endianness': 'LE',
  'shape': [64, 64],
  'order': 'C',
  'size': 0.016,          # BLOB size in MB
  'bits': 12,
  'resolution': 4096,
  'vmin': 0.0,
  'vmax': 1.0
}
```

### Python Low-Level (decode + unpack) 
```python
from imgpack import encode_data, pack_data, unpack_data, decode_data

blob, header = encode_data(arr, 0.0, 1.0, dtype="uint16", bits=12)
envelope = pack_data(blob, header)

header2, blob2 = unpack_data(envelope)
array2 = decode_data(header2, blob2)
```

### JavaScript (browser decode)

```js
import { decode } from "/decode.js";

ws.onmessage = async (evt) => {
  const ab = evt.data instanceof Blob ? await evt.data.arrayBuffer() : evt.data;
  const { header, typed } = decode(ab);
  renderGrayscale(canvas, typed, header);
};
```

---

## 📦 Binary Format

```
HEADER_LEN(4, BE) | HEADER_JSON(UTF-8) | PAD | BLOB
```

* **HEADER\_LEN**: JSON header size (uint32, big-endian)
* **HEADER\_JSON**: metadata (`format`, `version`, `dtype`, `shape`, `bits`, `resolution`, etc.)
* **PAD**: zero bytes to align BLOB start to element size (1, 2, 4, or 8)
* **BLOB**: raw packed/quantized/float data

---

## ⚠️ Limitations

* Only 2D arrays are supported right now.
* Quantization is **lossy** for `packed` and integer encodings (values outside `[vmin, vmax]` are clipped).
* For floats, values are stored losslessly but still aligned to chosen precision (`float16/32/64`).
