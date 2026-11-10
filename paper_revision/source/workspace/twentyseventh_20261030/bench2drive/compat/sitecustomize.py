"""Process-local Python3.10/NumPy legacy-name compatibility, no score changes.
Copy as sitecustomize.py in the isolated Bench2Drive path only.
"""
import collections,collections.abc
for name in ['Mapping','MutableMapping','Iterable','Sequence','MutableSequence']:
 if not hasattr(collections,name):setattr(collections,name,getattr(collections.abc,name))
import numpy as np
for name,value in [('int',int),('float',float),('bool',bool),('complex',complex),('object',object)]:
 if name not in np.__dict__:setattr(np,name,value)
