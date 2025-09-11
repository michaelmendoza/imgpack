// /imgpack/decode.js
// Envelope: HEADER_LEN(4, BE) | HEADER_JSON(utf-8) | PADDING | BLOB
// Exports: readHeaderAndBlob, blobToTypedArray, decode (combined)

export function readHeaderAndBlob(arrayBuffer) {
  const dv = new DataView(arrayBuffer);
  let off = 0;

  if (dv.byteLength < 4) throw new Error("Buffer too small");
  const headerLen = dv.getUint32(off, /*littleEndian=*/false);
  off += 4;
  if (off + headerLen > dv.byteLength) throw new Error("Invalid HEADER_LEN");

  const headerBytes = new Uint8Array(arrayBuffer, off, headerLen);
  off += headerLen;
  const headerStr = new TextDecoder("utf-8").decode(headerBytes);
  const header = JSON.parse(headerStr);

  const dtype = (header.dtype || "").toLowerCase();
  let elemSize = 1;
  if (dtype === "packed") elemSize = 1;
  else if (dtype === "uint8") elemSize = 1;
  else if (dtype === "uint16" || dtype === "float16") elemSize = 2;
  else if (dtype === "uint32" || dtype === "float32") elemSize = 4;
  else if (dtype === "float64") elemSize = 8;

  // Note: elemSize is always 1,2,4,8 => power-of-two, so this alignment trick is valid
  const padLen = ((-(4 + headerLen)) & (elemSize - 1)) >>> 0;
  off += padLen;

  const blobBytes = new Uint8Array(arrayBuffer, off);
  return { header, blobBytes };
}

export function unpackNBitsToTypedArray(u8, nbits, count) {
  const mask = (1 << nbits) - 1;
  let out;
  if (nbits <= 8) out = new Uint8Array(count);
  else if (nbits <= 16) out = new Uint16Array(count);
  else out = new Uint32Array(count);

  let acc = 0, accBits = 0, bi = 0;
  for (let i = 0; i < count; i++) {
    while (accBits < nbits) {
      if (bi >= u8.length) throw new Error("Packed data underrun");
      acc |= u8[bi] << accBits; accBits += 8; bi++;
    }
    out[i] = acc & mask;
    acc >>>= nbits; accBits -= nbits;
  }
  return out;
}

export function blobToTypedArray(header, blobBytes) {
  const [H, W] = header.shape;
  const count = H * W;
  const dtype = header.dtype.toLowerCase();

  if (dtype === "packed") {
    return unpackNBitsToTypedArray(blobBytes, header.bits, count);
  }

  if (dtype === "uint8")
    return new Uint8Array(blobBytes.buffer, blobBytes.byteOffset, blobBytes.byteLength);
  if (dtype === "uint16")
    return new Uint16Array(blobBytes.buffer, blobBytes.byteOffset, blobBytes.byteLength / 2);
  if (dtype === "uint32")
    return new Uint32Array(blobBytes.buffer, blobBytes.byteOffset, blobBytes.byteLength / 4);

  if (dtype === "float32")
    return new Float32Array(blobBytes.buffer, blobBytes.byteOffset, blobBytes.byteLength / 4);
  if (dtype === "float64")
    return new Float64Array(blobBytes.buffer, blobBytes.byteOffset, blobBytes.byteLength / 8);
  if (dtype === "float16") {
    if (typeof Float16Array !== "undefined") {
      return new Float16Array(blobBytes.buffer, blobBytes.byteOffset, blobBytes.byteLength / 2);
    } else {
      // Fallback: return the raw u16 view; app can convert to f32 if needed.
      return new Uint16Array(blobBytes.buffer, blobBytes.byteOffset, blobBytes.byteLength / 2);
    }
  }
  throw new Error(`Unsupported dtype ${dtype}`);
}

// High-level: decode = readHeaderAndBlob + blobToTypedArray
export function decode(arrayBuffer) {
  const { header, blobBytes } = readHeaderAndBlob(arrayBuffer);
  const typed = blobToTypedArray(header, blobBytes);
  return { header, typed };
}
