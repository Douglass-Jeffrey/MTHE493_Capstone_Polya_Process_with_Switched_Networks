"""ctypes wrapper for native/feeder.dll: worker threads reading file ranges into pinned buffers, by slot."""
import ctypes
import os

from .native import build as _build

ALIGN = 4096  # unbuffered reads need sector-aligned offsets, lengths and buffers


class Feeder:
    def __init__(self, threads=4, slots=8, chunk=8 << 20):
        self.lib = ctypes.CDLL(_build.build("feeder"))
        L = self.lib
        L.feeder_create.restype = ctypes.c_void_p
        L.feeder_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
        L.feeder_open.restype = ctypes.c_int
        L.feeder_open.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
        L.feeder_submit.restype = ctypes.c_int
        L.feeder_submit.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_longlong,
                                    ctypes.c_longlong, ctypes.c_void_p]
        L.feeder_submit_copy.restype = ctypes.c_int
        L.feeder_submit_copy.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_longlong,
                                         ctypes.c_void_p]
        L.feeder_wait.restype = ctypes.c_int
        L.feeder_wait.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.feeder_bytes.restype = ctypes.c_longlong
        L.feeder_bytes.argtypes = [ctypes.c_void_p]
        L.feeder_destroy.argtypes = [ctypes.c_void_p]
        assert chunk % ALIGN == 0
        self.h = L.feeder_create(threads, slots, chunk)
        self.files = {}

    def open(self, path, unbuffered=True):
        i = self.lib.feeder_open(self.h, os.path.abspath(path), int(unbuffered))
        if i < 0:
            raise OSError(-i, f"feeder_open failed for {path}")
        self.files[path] = i
        return i

    def submit(self, slot, file, off, length, dst_ptr):
        self.lib.feeder_submit(self.h, slot, file, off, length, dst_ptr)

    def submit_copy(self, slot, src_ptr, length, dst_ptr):
        self.lib.feeder_submit_copy(self.h, slot, src_ptr, length, dst_ptr)

    def wait(self, slot):
        e = self.lib.feeder_wait(self.h, slot)
        if e:
            raise OSError(e, f"read error in slot {slot}")

    @property
    def bytes_read(self):
        return self.lib.feeder_bytes(self.h)

    def close(self):
        if self.h:
            self.lib.feeder_destroy(self.h)
            self.h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
