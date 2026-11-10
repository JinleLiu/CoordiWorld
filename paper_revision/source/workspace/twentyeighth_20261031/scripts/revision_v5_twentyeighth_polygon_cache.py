"""Exact immutable polygon memo; keys preserve all six float64 bit patterns."""
from functools import lru_cache
import numpy as np
def install():
 import coordiworld.geometry as geometry
 original=geometry.box_polygon
 if hasattr(original,'exact_cache_info'):return original
 @lru_cache(maxsize=8192)
 def cached(key):
  x=np.frombuffer(key,dtype=np.float64);return original(x[:3],x[3],x[4],x[5])
 def wrapped(pose,length,width,rear_axle_offset=0.):
  return cached(np.asarray([*pose,length,width,rear_axle_offset],dtype=np.float64).tobytes())
 wrapped.exact_cache_info=cached.cache_info;wrapped.exact_cache_clear=cached.cache_clear;wrapped.original=original;geometry.box_polygon=wrapped;return wrapped
