"""Bound resource-only CUDA graph caches; leave original numerical code intact."""
from collections import OrderedDict
import gc
import hashlib
import inspect
from pathlib import Path
import torch

WORLD_SOURCE_SHA256 = '8a7c3b806ffe3b50f2a7eebb4af6f8ea2e84d08162d5b90cee73b22b8abd2a17'


class BoundedGraphCache:
    def __init__(self, limit=2):
        self.entries = OrderedDict()
        self.limit = limit
        self.evictions = 0
        self.high_water = 0

    def __contains__(self, key):
        present = key in self.entries
        if not present and len(self) >= self.limit:
            # Membership is tested BEFORE CapturedFactual/CapturedRail construction.
            # Evicting at __setitem__ would leave both graphs alive at peak allocation.
            torch.cuda.synchronize()
            _, graph = self.entries.popitem(last=False)
            del graph
            gc.collect()
            torch.cuda.empty_cache()
            self.evictions += 1
        return present

    def __getitem__(self, key):
        graph = self.entries[key]
        self.entries.move_to_end(key)
        return graph

    def __setitem__(self, key, graph):
        assert key in self.entries or len(self) < self.limit
        self.entries[key] = graph
        self.entries.move_to_end(key)
        self.high_water = max(self.high_water, len(self))

    def receipt(self):
        return dict(entries=len(self), limit=self.limit, evictions=self.evictions,
                    high_water=self.high_water)

    def __len__(self):
        return len(self.entries)


def install(pipe, limit=2):
    source = Path(inspect.getfile(type(pipe.world)))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == WORLD_SOURCE_SHA256
    for name in ['graphs', 'factual_graphs']:
        old = getattr(pipe.world, name)
        assert type(old) is dict and not old, 'Install before first inference'
        setattr(pipe.world, name, BoundedGraphCache(limit))
    return dict(source=str(source), source_sha256=WORLD_SOURCE_SHA256,
                max_entries_per_cache=limit, cache_names=['graphs','factual_graphs'],
                numerical_source_modified=False,
                method='LRU eviction before graph construction; synchronize and release dead graph pools')


def status(pipe):
    return {name:getattr(pipe.world, name).receipt() for name in ['graphs','factual_graphs']}
