"""What Kiln returns that is not plain JSON: arrays of numbers and images."""
import array
import struct
import sys
import zlib

# Kiln's array types and the array module's codes for them.
TYPECODES = {
    "float32": "f",
    "float64": "d",
    "int8": "b",
    "uint8": "B",
    "int16": "h",
    "uint16": "H",
    "int32": "i",
    "uint32": "I",
    "int64": "q",
    "uint64": "Q",
}


class KilnError(Exception):
    """An error from Kiln, from Node.js, or from starting them."""


def _native(data, dtype):
    """The values of little-endian `data` as an array.array."""
    a = array.array(TYPECODES[dtype])
    a.frombytes(bytes(data))
    if sys.byteorder == "big":
        a.byteswap()
    return a


class Tensor:
    """Numbers with a shape: `dims`, `dtype` and the values as little-endian bytes in `data`.

    `tolist()` gives nested lists, `numpy()` a NumPy array (NumPy is not needed otherwise), and
    `numpy.asarray(tensor)` works too.
    """

    def __init__(self, dtype, dims, data):
        if dtype not in TYPECODES:
            raise KilnError("unknown array type %r" % (dtype,))
        self.dtype = dtype
        self.dims = [int(d) for d in dims]
        self.data = bytes(data)

    @property
    def shape(self):
        return tuple(self.dims)

    @property
    def size(self):
        return len(self.data) // struct.calcsize(TYPECODES[self.dtype])

    def __len__(self):
        return self.dims[0] if self.dims else 0

    def tolist(self):
        flat = _native(self.data, self.dtype).tolist()

        def rec(offset, d):
            n = self.dims[d]
            if d == len(self.dims) - 1:
                return flat[offset : offset + n]
            stride = 1
            for x in self.dims[d + 1 :]:
                stride *= x
            return [rec(offset + i * stride, d + 1) for i in range(n)]

        return rec(0, 0) if self.dims else (flat[0] if flat else None)

    def numpy(self):
        try:
            import numpy as np
        except ImportError:
            raise KilnError("Tensor.numpy() needs NumPy (pip install numpy); tolist() works without it")
        return np.frombuffer(self.data, dtype=np.dtype(self.dtype).newbyteorder("<")).reshape(self.dims)

    def __array__(self, dtype=None, copy=None):
        a = self.numpy()
        return a.astype(dtype) if dtype is not None else a

    def __repr__(self):
        return "Tensor(dtype=%r, dims=%r)" % (self.dtype, self.dims)


def _png(width, height, channels, data):
    """`data` (rows of 8-bit gray, RGB or RGBA) as the bytes of a PNG file."""
    color = {1: 0, 2: 4, 3: 2, 4: 6}[channels]
    row = width * channels
    raw = b"".join(b"\x00" + data[y * row : (y + 1) * row] for y in range(height))

    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


class RawImage:
    """Pixels: `width`, `height`, `channels` (1 gray, 3 RGB, 4 RGBA) and `data`, one byte per channel, row by row."""

    def __init__(self, width, height, channels, data):
        self.width = int(width)
        self.height = int(height)
        self.channels = int(channels)
        self.data = bytes(data)
        if len(self.data) != self.width * self.height * self.channels:
            raise KilnError("an image of %dx%d with %d channels has %d bytes, got %d" % (self.width, self.height, self.channels, self.width * self.height * self.channels, len(self.data)))

    def numpy(self):
        """A NumPy array [height, width, channels] ([height, width] for one channel)."""
        try:
            import numpy as np
        except ImportError:
            raise KilnError("RawImage.numpy() needs NumPy (pip install numpy)")
        a = np.frombuffer(self.data, dtype=np.uint8)
        return a.reshape(self.height, self.width) if self.channels == 1 else a.reshape(self.height, self.width, self.channels)

    def to_pil(self):
        try:
            from PIL import Image
        except ImportError:
            raise KilnError("RawImage.to_pil() needs Pillow (pip install pillow); save() writes a PNG without it")
        return Image.frombytes({1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}[self.channels], (self.width, self.height), self.data)

    def save(self, path):
        """Writes the image: a .png with nothing else installed, other formats through Pillow."""
        path = str(path)
        if path.lower().endswith(".png"):
            with open(path, "wb") as f:
                f.write(_png(self.width, self.height, self.channels, self.data))
        else:
            self.to_pil().save(path)

    def __repr__(self):
        return "RawImage(width=%d, height=%d, channels=%d)" % (self.width, self.height, self.channels)


def write_wav(path, samples, sampling_rate):
    """Writes mono samples in [-1, 1] (a Tensor, a list or a NumPy array) as a 16-bit WAV file."""
    import wave

    if isinstance(samples, Tensor):
        values = _native(samples.data, samples.dtype)
    elif hasattr(samples, "tolist"):
        values = samples.tolist()
    else:
        values = samples
    pcm = array.array("h", (max(-32768, min(32767, int(round(float(x) * 32767)))) for x in values))
    if sys.byteorder == "big":
        pcm.byteswap()
    w = wave.open(str(path), "wb")
    try:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sampling_rate))
        w.writeframes(pcm.tobytes())
    finally:
        w.close()
