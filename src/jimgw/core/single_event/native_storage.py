"""Temporary file-backed native arrays with explicit bounded page residency.

The file handle lives with the Data instance; NumPy views retain the mapping.
Dropping clean mapped pages is a cache hint, not deletion of the native data.
Explicit array conversion or time-domain operations may still allocate memory.
"""

import mmap
import tempfile

import jax
import numpy as np


def allocate_native_strain(n_frequencies: int, storage: str = "memory"):
    """Return (complex128 array, owner) without a second full-length buffer."""
    if storage == "memory":
        return np.zeros(n_frequencies, dtype=np.complex128), None
    if storage != "mmap":
        raise ValueError("host_storage must be memory or mmap")
    # Ownership transfers to Data and must outlive this function's scope.
    owner = tempfile.TemporaryFile(prefix="jimgw-native-", suffix=".fd")  # noqa: SIM115
    n_bytes = n_frequencies * np.dtype(np.complex128).itemsize
    try:
        owner.truncate(n_bytes)
        mapping = mmap.mmap(owner.fileno(), n_bytes, access=mmap.ACCESS_WRITE)
    except BaseException:
        owner.close()
        raise
    # Newly extended file bytes are zero, including frequencies outside the
    # analysis band. No full-array memset or ndarray-to-memmap copy is needed.
    return np.ndarray((n_frequencies,), dtype=np.complex128, buffer=mapping), owner


def mapped_region(array):
    """Return the backing mapping and byte range for a contiguous NumPy view."""
    if (
        not isinstance(array, np.ndarray)
        or not array.flags.c_contiguous
        or not array.size
    ):
        return None
    base = array
    while isinstance(base, np.ndarray):
        base = base.base
    if not isinstance(base, mmap.mmap):
        return None
    origin = np.frombuffer(base, dtype=np.uint8, count=1).ctypes.data
    start = array.ctypes.data - origin
    end = start + array.nbytes
    if start < 0 or end > len(base):
        raise ValueError("Native array view extends outside its mapping")
    return base, start, end


def release_mapped_pages(array, *, written: bool = False) -> bool:
    """Flush a written slice and advise eviction of its clean mapped pages.

    Call after a device transfer has completed. The advice is available on
    Unix platforms exposing MADV_DONTNEED; elsewhere file backing still works
    but residency is left to the OS. The leading page may include a previously
    completed chunk; dropping it prevents boundary pages accumulating when
    chunks are not page-aligned. The trailing partial page waits for the next
    chunk unless it is the final page of the mapping.
    """
    region = mapped_region(array)
    if region is None:
        return False
    mapping, start, end = region
    page = mmap.PAGESIZE
    if written:
        flush_start = start // page * page
        mapping.flush(flush_start, end - flush_start)
    if not hasattr(mapping, "madvise") or not hasattr(mmap, "MADV_DONTNEED"):
        return False
    drop_start = start // page * page
    drop_end = end if end == len(mapping) else end // page * page
    if drop_end > drop_start:
        mapping.madvise(mmap.MADV_DONTNEED, drop_start, drop_end - drop_start)
    return True


def native_storage_accounting(detectors) -> dict[str, int]:
    """Count existing native buffers without copying or materializing data.

    Host allocations and mappings are deduplicated by their backing owner, so
    shared grids, band views, and scalar broadcast windows count only once.
    Host allocation bytes describe retained buffers, not process RSS; mapped
    file bytes are logical storage and do not imply resident pages. Device
    arrays are deduplicated by object identity only: their logical sizes are
    not measurements of device allocator usage or cross-object aliasing.
    Buffers retained only by external callers are outside this accounting.
    """
    counts = {
        "host_allocation_count": 0,
        "host_allocation_bytes": 0,
        "mapped_file_count": 0,
        "mapped_file_logical_bytes": 0,
        "device_array_object_count": 0,
        "device_array_logical_bytes": 0,
        "unclassified_array_object_count": 0,
    }
    seen_host = set()
    seen_mapped = set()
    seen_device = set()
    seen_unclassified = set()

    def add_array(array):
        if array is None:
            return
        if isinstance(array, jax.Array):
            if id(array) not in seen_device:
                seen_device.add(id(array))
                counts["device_array_object_count"] += 1
                counts["device_array_logical_bytes"] += int(array.nbytes)
            return
        owner = array
        while True:
            if isinstance(owner, np.ndarray) and owner.base is not None:
                owner = owner.base
            elif isinstance(owner, memoryview):
                owner = owner.obj
            else:
                break
        if isinstance(owner, mmap.mmap):
            if id(owner) not in seen_mapped:
                seen_mapped.add(id(owner))
                counts["mapped_file_count"] += 1
                counts["mapped_file_logical_bytes"] += len(owner)
        elif isinstance(owner, (np.ndarray, bytes, bytearray)):
            if id(owner) not in seen_host:
                seen_host.add(id(owner))
                counts["host_allocation_count"] += 1
                counts["host_allocation_bytes"] += (
                    int(owner.nbytes) if isinstance(owner, np.ndarray) else len(owner)
                )
        elif id(owner) not in seen_unclassified:
            seen_unclassified.add(id(owner))
            counts["unclassified_array_object_count"] += 1

    for detector in detectors:
        for array in detector.data.materialized_arrays():
            add_array(array)
        psd = getattr(detector, "psd", None)
        for name in ("frequencies", "values"):
            add_array(getattr(psd, name, None))
    return counts


__all__ = [
    "allocate_native_strain",
    "mapped_region",
    "native_storage_accounting",
    "release_mapped_pages",
]
