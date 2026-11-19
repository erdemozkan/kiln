"""Python values to and from the bridge's frames (see _js/bridge.mjs for the format)."""
import array
import json
import math
import os
import struct
import sys

from ._types import TYPECODES, KilnError, RawImage, Tensor

IMAGE_FILES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")
_CODES = {}
for _name, _code in TYPECODES.items():
    _CODES[_code] = _name
# The array module's "l" and "L" are 4 or 8 bytes by platform.
_CODES["l"] = "int32" if array.array("l").itemsize == 4 else "int64"
_CODES["L"] = "uint32" if array.array("L").itemsize == 4 else "uint64"


def _blob(blobs, data, dtype):
    blobs.append(data)
    return {"$blob": len(blobs) - 1, "dtype": dtype}


def _pixels(blobs, width, height, data):
    # Kiln takes RGB or RGBA and tells them apart by the length.
    return {"width": width, "height": height, "data": _blob(blobs, data, "uint8")}


def _pil(blobs, img):
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if "A" in img.mode or "transparency" in getattr(img, "info", {}) else "RGB")
    return _pixels(blobs, img.size[0], img.size[1], img.tobytes())


def read_image_file(path):
    """An image file as a Pillow image (Kiln has no image decoders outside a browser)."""
    try:
        from PIL import Image
    except ImportError:
        raise KilnError("reading %s needs Pillow (pip install pillow); or give the pixels as a NumPy array or a RawImage" % (path,))
    img = Image.open(path)
    img.load()
    return img


def _numpy(blobs, a):
    import numpy as np

    a = np.asarray(a)
    if a.dtype == np.uint8 and a.ndim == 3 and a.shape[2] in (1, 3, 4):
        # [height, width, channels] of bytes is an image.
        if a.shape[2] == 1:
            a = np.repeat(a, 3, axis=2)
        return _pixels(blobs, int(a.shape[1]), int(a.shape[0]), np.ascontiguousarray(a).tobytes())
    if a.dtype == np.bool_:
        a = a.astype(np.uint8)
    elif a.dtype.kind == "f" and a.dtype.itemsize != 4:
        # Kiln computes in 32-bit floats.
        a = a.astype(np.float32)
    name = a.dtype.name
    if name not in TYPECODES:
        raise KilnError("NumPy arrays of %s do not go to Kiln" % (name,))
    return _blob(blobs, np.ascontiguousarray(a.astype(a.dtype.newbyteorder("<"), copy=False)).tobytes(), name)


def encode(v, blobs, callbacks):
    """`v` as JSON for a frame's header: bytes and arrays go to `blobs`, functions to `callbacks`."""
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, int):
        return v if -(2**53) < v < 2**53 else {"$bigint": str(v)}
    if isinstance(v, float):
        if math.isfinite(v):
            return v
        return {"$float": "NaN" if math.isnan(v) else ("Infinity" if v > 0 else "-Infinity")}
    if isinstance(v, dict):
        return {str(k): encode(x, blobs, callbacks) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [encode(x, blobs, callbacks) for x in v]
    if isinstance(v, (bytes, bytearray, memoryview)):
        return _blob(blobs, bytes(v), "uint8")
    if isinstance(v, array.array):
        if v.typecode not in _CODES:
            raise KilnError("array.array of type %r does not go to Kiln" % (v.typecode,))
        if sys.byteorder == "big":
            v = array.array(v.typecode, v)
            v.byteswap()
        return _blob(blobs, v.tobytes(), _CODES[v.typecode])
    if isinstance(v, Tensor):
        return _blob(blobs, v.data, v.dtype)
    if isinstance(v, RawImage):
        if v.channels not in (3, 4):
            raise KilnError("Kiln takes RGB or RGBA images, this one has %d channels" % (v.channels,))
        return _pixels(blobs, v.width, v.height, v.data)
    if isinstance(v, os.PathLike):
        path = os.fspath(v)
        if path.lower().endswith(IMAGE_FILES):
            return _pil(blobs, read_image_file(path))
        with open(path, "rb") as f:
            return _blob(blobs, f.read(), "uint8")
    if hasattr(v, "__array_interface__"):
        return _numpy(blobs, v)
    if hasattr(v, "mode") and hasattr(v, "size") and hasattr(v, "tobytes") and hasattr(v, "convert"):
        return _pil(blobs, v)
    if hasattr(v, "item") and hasattr(v, "dtype"):
        return encode(v.item(), blobs, callbacks)
    if callable(v):
        callbacks.append(v)
        return {"$callback": len(callbacks) - 1}
    raise KilnError("a %s does not go to Kiln" % (type(v).__name__,))


def decode(v, blobs):
    """A frame's JSON back as Python values: arrays as Tensors, pixels as RawImages."""
    if isinstance(v, list):
        return [decode(x, blobs) for x in v]
    if not isinstance(v, dict):
        return v
    # The tags are read only in the exact form Kiln writes them: text a model wrote (a tool call's
    # arguments) can hold these keys too, and then it is a plain dict.
    keys = set(v)
    index = v.get("$blob")
    if keys == {"$blob", "dtype"} and type(index) is int and 0 <= index < len(blobs) and v["dtype"] in TYPECODES:
        data = blobs[index]
        return Tensor(v["dtype"], [len(data) // struct.calcsize(TYPECODES[v["dtype"]])], data)
    if keys == {"$bigint"} and isinstance(v["$bigint"], str) and len(v["$bigint"]) <= 400 and v["$bigint"].lstrip("-").isdigit():
        return int(v["$bigint"])
    if keys == {"$float"} and v["$float"] in ("NaN", "Infinity", "-Infinity"):
        return float(v["$float"].replace("Infinity", "inf"))
    d = {k: decode(x, blobs) for k, x in v.items()}
    data = d.get("data")
    if isinstance(data, Tensor):
        if set(d) == {"type", "data", "dims"}:
            return Tensor(data.dtype, d["dims"], data.data)
        if data.dtype == "uint8" and set(d) in ({"width", "height", "data"}, {"width", "height", "channels", "data"}):
            pixels = d["width"] * d["height"]
            return RawImage(d["width"], d["height"], d.get("channels") or (len(data.data) // pixels if pixels else 0), data.data)
    return d


def pack(header, blobs):
    """A frame's bytes: length, JSON header, blobs."""
    header = dict(header)
    header["blobs"] = [len(b) for b in blobs]
    body = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return b"".join([struct.pack("<I", len(body)), body] + list(blobs))


def unpack(read):
    """Reads one frame with `read(n)` (n bytes, fewer only at the end); None when there is nothing left."""
    head = read(4)
    if len(head) < 4:
        return None
    (n,) = struct.unpack("<I", head)
    body = read(n)
    if len(body) < n:
        return None
    header = json.loads(body.decode("utf-8"))
    blobs = []
    for size in header.get("blobs", []):
        b = read(size)
        if len(b) < size:
            return None
        blobs.append(b)
    return header, blobs
