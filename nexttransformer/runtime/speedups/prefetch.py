"""Pack the next rows on a background thread while the GPU trains on the current ones.

The wrapped stream is deterministic and only this thread touches it, so rows, their order and the
checkpointed cursor are exactly those of the synchronous stream. `state_dict` reports the state
after the last batch the trainer consumed, not the prefetch position: documents are never mutated,
so a shallow copy of the buffer per batch is a complete snapshot.
"""
import copy
import queue
import threading

import torch


class PrefetchedStream:
    def __init__(self, stream, depth=8):
        self.stream, self.depth = stream, depth
        self.thread, self.size, self.error = None, None, None
        self.ready = queue.Queue(depth)
        self.consumed = self._snapshot()

    @property
    def rows(self): return self.consumed["rows"]

    def _snapshot(self):
        s = self.stream
        return dict(identity=s.identity, cursor=dict(s.cursor), buffer=list(s.buffer), rows=s.rows)

    def _produce(self, device):
        try:
            torch.cuda.set_device(device)
            while True:
                rows = torch.tensor([self.stream._row() for _ in range(self.size)], dtype=torch.long)
                x, y = rows[:, :-1].contiguous().pin_memory(), rows[:, 1:].contiguous().pin_memory()
                self.ready.put((x, y, self._snapshot()))
        except BaseException as exc:
            self.error = exc
            self.ready.put(None)

    def batch(self, batch_size, device):
        if device.type != "cuda":
            raise RuntimeError("the prefetched stream serves CUDA batches of one fixed size")
        if self.thread is None:
            self.size = batch_size
            self.thread = threading.Thread(target=self._produce, args=(device,), name="row-prefetch", daemon=True)
            self.thread.start()
        if batch_size != self.size: raise RuntimeError("the prefetched stream serves one batch size")
        item = self.ready.get()
        if item is None: raise RuntimeError("row prefetch failed") from self.error
        x, y, self.consumed = item
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    def state_dict(self):
        return copy.deepcopy(self.consumed)

    def load_state_dict(self, state):
        if self.thread is not None: raise RuntimeError("load the stream state before training starts")
        self.stream.load_state_dict(state)
        self.consumed = self._snapshot()
