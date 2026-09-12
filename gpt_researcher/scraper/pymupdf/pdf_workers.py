"""Bounded, disposable PDF processes. No PDF library runs in caller threads."""

from collections import deque
from dataclasses import dataclass, field
import multiprocessing
import os
import queue
import threading
import time


MAX_WORKERS = max(1, min(4, os.cpu_count() or 1))
_LIGHT_TIMEOUT_SECONDS = 30
_DOCLING_TIMEOUT_SECONDS = 120
# ponytail: per-process caps; use deployment quotas to bound multiple replicas.
_slots = threading.BoundedSemaphore(MAX_WORKERS)
_ocr_slots = threading.BoundedSemaphore(min(2, MAX_WORKERS))


@dataclass(frozen=True)
class Batch:
    route: str
    indices: tuple[int, ...] = ()
    force_ocr: bool = False
    retry: int = 0


@dataclass
class _Job:
    batch: Batch
    process: object
    reader: threading.Thread
    messages: queue.SimpleQueue
    budget: float
    expires: float
    active_index: int | None = None
    started: set[int] = field(default_factory=set)
    completed: set[int] = field(default_factory=set)


def _receive(connection, messages):
    # Receiving a partial pipe frame must not block the supervising thread.
    try:
        while True:
            message = connection.recv()
            messages.put((time.monotonic(), message))
    except (EOFError, OSError):
        pass
    finally:
        connection.close()


def _stop(process):
    if process.is_alive():
        process.terminate()
    process.join(1)
    if process.is_alive():
        process.kill()
        process.join()
    process.close()


def run_batches(pdf_path, pending: deque, deadline, target):
    """Yield (batch, message), allowing the caller to enqueue escalation.

    A final ('finished',) follows received messages, even after a crash.
    Closing the generator reaps every owned process before returning.
    """
    context = multiprocessing.get_context("spawn")
    active = []

    def dispose(job):
        _stop(job.process)
        job.reader.join()
        _slots.release()
        if job.batch.route == "docling":
            _ocr_slots.release()

    def drain(job):
        while not job.messages.empty():
            received, message = job.messages.get()
            kind = message[0] if isinstance(message, tuple) and message else None
            if kind == "started":
                index = message[1] if len(message) > 1 else None
                position = len(job.started)
                expected = (job.batch.indices[position]
                            if position < len(job.batch.indices) else None)
                if (received <= job.expires and index == expected
                        and job.active_index is None):
                    job.started.add(index)
                    job.active_index = index
                    job.expires = min(deadline, received + job.budget)
            elif kind == "page" and len(message) > 1:
                index = getattr(message[1], "index", None)
                if (received <= job.expires and index == job.active_index
                        and index not in job.completed):
                    job.completed.add(index)
                    job.active_index = None
                elif index in job.batch.indices:
                    continue
            yield message

    try:
        while pending or active:
            if time.monotonic() >= deadline:
                pending.clear()
            for _ in range(len(pending)):
                batch = pending.popleft()
                heavy = batch.route == "docling"
                if heavy and not _ocr_slots.acquire(blocking=False):
                    pending.append(batch)
                    continue
                if not _slots.acquire(blocking=False):
                    if heavy:
                        _ocr_slots.release()
                    pending.appendleft(batch)
                    break
                receive = send = process = None
                try:
                    receive, send = context.Pipe(duplex=False)
                    messages = queue.SimpleQueue()
                    process = context.Process(target=target, args=(send, pdf_path, batch))
                    process.start()
                    send.close()
                    reader = threading.Thread(target=_receive, args=(receive, messages))
                    reader.start()
                except Exception:
                    if process is not None:
                        if process.pid is not None:
                            _stop(process)
                        else:
                            process.close()
                    if receive is not None:
                        receive.close()
                    if send is not None:
                        send.close()
                    _slots.release()
                    if heavy:
                        _ocr_slots.release()
                    yield batch, ("finished",)
                    continue
                budget = (_DOCLING_TIMEOUT_SECONDS if heavy
                          else _LIGHT_TIMEOUT_SECONDS)
                active.append(_Job(batch, process, reader, messages, budget,
                                   min(deadline, time.monotonic() + budget)))

            for job in active[:]:
                yield from ((job.batch, message) for message in drain(job))
                finished = (not job.process.is_alive()
                            or time.monotonic() >= job.expires)
                if finished:
                    dispose(job)
                    active.remove(job)
                    yield from ((job.batch, message) for message in drain(job))
                if finished:
                    yield job.batch, ("finished",)
            if pending or active:
                time.sleep(0.01)
    finally:
        for job in active:
            dispose(job)
