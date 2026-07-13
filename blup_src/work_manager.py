from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Lock
from typing import Callable, Hashable, Protocol


DisplayCallback = Callable[[], None]
DisplayScheduler = Callable[[DisplayCallback], None]
ErrorHandler = Callable[[str, int, BaseException], None]


@dataclass(frozen=True)
class WorkJob:
    id: int
    payload: object


@dataclass(frozen=True)
class WorkResult:
    id: int
    payload: object


class WorkRequest(Protocol):
    scope_key: str
    request_key: Hashable

    def make_jobs(self) -> list[WorkJob]:
        ...

    def begin_apply(self) -> None:
        ...

    def run_job(self, job: WorkJob) -> WorkResult:
        ...

    def apply_result(self, result: WorkResult) -> None:
        ...

    def finish_apply(self, *, cancelled: bool) -> None:
        ...


class WorkRequestBase:
    scope_key: str
    request_key: Hashable

    def begin_apply(self) -> None:
        return

    def finish_apply(self, *, cancelled: bool) -> None:
        return


@dataclass
class ActiveRequest:
    request_id: int
    scope_key: str
    request_key: Hashable
    request: WorkRequest
    pending_jobs: int
    cancelled: bool = False


@dataclass(frozen=True)
class ResultEnvelope:
    request_id: int
    scope_key: str
    result: WorkResult | None = None
    error: BaseException | None = None


class WorkManager:
    def __init__(
        self,
        *,
        schedule_display_callback: DisplayScheduler,
        max_workers: int = 4,
        error_handler: ErrorHandler | None = None,
    ) -> None:
        self._schedule_display_callback = schedule_display_callback
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._error_handler = error_handler or self.default_error_handler

        self._lock = Lock()
        self._next_request_id = 1
        self._active_by_scope: dict[str, ActiveRequest] = {}

        self._result_queue: Queue[ResultEnvelope] = Queue()
        self._drain_scheduled = False

    # -------------------------------------------
    # |               Public API                |
    # -------------------------------------------

    def submit(self, request: WorkRequest) -> int:
        jobs = list(request.make_jobs())

        cancelled_request: WorkRequest | None = None
        cancelled_scope: str | None = None
        request_id: int

        with self._lock:
            current = self._active_by_scope.get(request.scope_key)
            if (
                current is not None
                and not current.cancelled
                and current.request_key == request.request_key
            ):
                return current.request_id

            if current is not None and not current.cancelled:
                current.cancelled = True
                cancelled_request = current.request
                cancelled_scope = current.scope_key

            request_id = self._next_request_id
            self._next_request_id += 1

            state = ActiveRequest(
                request_id=request_id,
                scope_key=request.scope_key,
                request_key=request.request_key,
                request=request,
                pending_jobs=len(jobs),
            )
            self._active_by_scope[request.scope_key] = state

        if cancelled_request is not None and cancelled_scope is not None:
            cancelled_request.finish_apply(cancelled=True)

        request.begin_apply()

        if not jobs:
            self.finish_if_current(scope_key=request.scope_key, request_id=request_id, cancelled=False)
            return request_id

        for job in jobs:
            self._executor.submit(self.run_job, request_id, request, job)

        return request_id

    def cancel_scope(self, scope_key: str) -> None:
        cancelled_request: WorkRequest | None = None

        with self._lock:
            current = self._active_by_scope.pop(scope_key, None)
            if current is not None and not current.cancelled:
                current.cancelled = True
                cancelled_request = current.request

        if cancelled_request is not None:
            cancelled_request.finish_apply(cancelled=True)

    def shutdown(self, *, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=False)

    # -------------------------------------------
    # | Worker path |
    # -------------------------------------------

    def run_job(self, request_id: int, request: WorkRequest, job: WorkJob) -> None:
        try:
            result = request.run_job(job)
            envelope = ResultEnvelope(
                request_id=request_id,
                scope_key=request.scope_key,
                result=result,
                error=None,
            )
        except BaseException as exc:
            envelope = ResultEnvelope(
                request_id=request_id,
                scope_key=request.scope_key,
                result=None,
                error=exc,
            )

        self._result_queue.put(envelope)
        self.schedule_drain()

    # -------------------------------------------
    # | Display-thread path |
    # -------------------------------------------

    def drain_results(self) -> None:
        while True:
            try:
                envelope = self._result_queue.get_nowait()
            except Empty:
                break
            self.apply_envelope(envelope)

        reschedule = False
        with self._lock:
            self._drain_scheduled = False
            if not self._result_queue.empty():
                self._drain_scheduled = True
                reschedule = True

        if reschedule:
            self._schedule_display_callback(self.drain_results)

    def apply_envelope(self, envelope: ResultEnvelope) -> None:
        with self._lock:
            current = self._active_by_scope.get(envelope.scope_key)
            if current is None:
                return
            if current.request_id != envelope.request_id:
                return
            if current.cancelled:
                return
            request = current.request

        if envelope.error is not None:
            self._error_handler(envelope.scope_key, envelope.request_id, envelope.error)
        else:
            assert envelope.result is not None
            request.apply_result(envelope.result)

        self.mark_job_complete(envelope.scope_key, envelope.request_id)

    def mark_job_complete(self, scope_key: str, request_id: int) -> None:
        finished_request: WorkRequest | None = None

        with self._lock:
            current = self._active_by_scope.get(scope_key)
            if current is None:
                return
            if current.request_id != request_id:
                return
            if current.cancelled:
                return

            current.pending_jobs -= 1
            if current.pending_jobs > 0:
                return

            finished_request = current.request
            self._active_by_scope.pop(scope_key, None)

        if finished_request is not None:
            finished_request.finish_apply(cancelled=False)

    def finish_if_current(self, *, scope_key: str, request_id: int, cancelled: bool) -> None:
        finished_request: WorkRequest | None = None

        with self._lock:
            current = self._active_by_scope.get(scope_key)
            if current is None:
                return
            if current.request_id != request_id:
                return

            finished_request = current.request
            self._active_by_scope.pop(scope_key, None)

        if finished_request is not None:
            finished_request.finish_apply(cancelled=cancelled)

    def schedule_drain(self) -> None:
        should_schedule = False

        with self._lock:
            if not self._drain_scheduled:
                self._drain_scheduled = True
                should_schedule = True

        if should_schedule:
            self._schedule_display_callback(self.drain_results)

    @staticmethod
    def default_error_handler(scope_key: str, request_id: int, exc: BaseException) -> None:
        print(f"[WorkManager] scope={scope_key!r} request_id={request_id} error={exc!r}")
