from __future__ import annotations

import heapq
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Lock
from typing import Callable, Hashable, Literal, Protocol, TYPE_CHECKING, cast

from bokeh.models.layouts import LayoutDOM

from blup.state import ModuleID

if TYPE_CHECKING:
    from blup.controller import AppController


RequestPriority = int
RequestKind = Literal["update", "background"]
ExecutionKind = Literal["thread", "process"]
DispatchLane = Literal["reserved", "shared", "process"]

UPDATE_PRIORITY: RequestPriority = 0
BACKGROUND_PRIORITY: RequestPriority = 10

DisplayCallback = Callable[[], None]
DisplayScheduler = Callable[[DisplayCallback], None]
ErrorHandler = Callable[[str, int, BaseException], None]

# -------------------------------------------
# |            Module Protocols             |
# -------------------------------------------

class RequestSpec(Protocol):
    @property
    def request_key(self) -> Hashable: ...


class Surface(Protocol):
    @property
    def root(self) -> LayoutDOM | None: ...

    def build(self) -> LayoutDOM:
        ...


@dataclass(frozen=True)
class WorkJob[JobT]:
    id:         int
    payload:    JobT


class Assembler[UpdT, ReqT: RequestSpec, JobT, ResT](Protocol):

    def prepare_request(self, update: UpdT) -> ReqT:
        ...
    # NOTE: possibly remove later; update RequestSpec to always include jobs
    def unroll_job_list(self, request: ReqT) -> list[WorkJob[JobT]]:
        ...
    def run_job(self, job: WorkJob[JobT]) -> ResT:
        ...


class Pipeline(Protocol):
    @property
    def module_id(self) -> ModuleID: ...
    @property
    def root(self) -> LayoutDOM | None: ...

    @property
    def subscribed_state(self) -> tuple[str, ...]: ...

    # controller facing interface:
    def build(self) -> LayoutDOM:
        ...
    def bind(self, host: "AppController") -> None:
        ...
    def refresh(self, host: "AppController") -> None:
        ...


class WorkPipeline[UpdT, ReqT: RequestSpec, JobT, ResT](Pipeline, Protocol):
    @property
    def assembler(self) -> Assembler[UpdT, ReqT, JobT, ResT]: ...

    # async update interface:
    def prepare_update(self, host: "AppController") -> UpdT:
        ...
    def start_update(self) -> None:
        ...
    def apply_result(self, result: ResT) -> None:
        ...
    def finish_update(self, *, cancelled: bool) -> None:
        ...

# -------------------------------------------
# |              Work Manager               |
# -------------------------------------------

# external work request contracts
# ---------------------

# For module UI update requests:
class UIWorkRequest[UpdT, ReqT: RequestSpec, JobT, ResT]:
    request:        ReqT
    pipeline:       WorkPipeline[UpdT, ReqT, JobT, ResT]

    scope_key:      str
    request_key:    Hashable
    priority:       RequestPriority
    request_kind:   RequestKind
    exec_kind:      ExecutionKind

    def __init__(
        self,
        *,
        request: ReqT,
        pipeline: WorkPipeline[UpdT, ReqT, JobT, ResT],
        scope_key: str | None = None,
        priority: RequestPriority = UPDATE_PRIORITY,
        exec_kind: ExecutionKind = "thread",
    ) -> None:
        self.request = request
        self.pipeline = pipeline

        self.scope_key = scope_key or pipeline.module_id
        self.request_key = request.request_key
        self.priority = priority
        self.request_kind = "update"
        self.exec_kind = exec_kind

    # generic request lifecycle
    def unroll_job_list(self) -> list[object]:
        jobs = self.pipeline.assembler.unroll_job_list(self.request)
        return cast(list[object], jobs)

    def on_start_hook(self) -> None:
        self.pipeline.start_update()

    def run_job(self, job: object) -> object:
        typed_job = cast(WorkJob[JobT], job)
        return self.pipeline.assembler.run_job(typed_job)

    def on_result_hook(self, result: object) -> None:
        typed_res = cast(ResT, result)
        self.pipeline.apply_result(typed_res)

    def on_finish_hook(self, *, cancelled: bool) -> None:
        self.pipeline.finish_update(cancelled=cancelled)


# For background data pre-fetching requests:
class BGWorkRequest[UpdT, CtxT, ReqT: RequestSpec, JobT, ResT]:
    request:        ReqT
    # NOTE: for now this is an assembler object
    # but this will likely change when prefetch
    # is fully implemented; semantics will be
    # similar to the current assembler
    assembler:      Assembler[UpdT, ReqT, JobT, ResT]

    scope_key:      str
    request_key:    Hashable
    priority:       RequestPriority
    request_kind:   RequestKind
    exec_kind:      ExecutionKind

    def __init__(
        self,
        *,
        request: ReqT,
        assembler: Assembler[UpdT, ReqT, JobT, ResT],
        scope_key: str,
        priority: RequestPriority = BACKGROUND_PRIORITY,
        exec_kind: ExecutionKind = "thread",
        on_start: Callable[[ReqT], None] | None = None,
        on_result: Callable[[ResT], None] | None = None,
        on_finish: Callable[[bool], None] | None = None,
    ) -> None:
        self.request = request
        self.assembler = assembler

        self.scope_key = scope_key
        self.request_key = request.request_key
        self.priority = int(priority)
        self.request_kind = "background"
        self.exec_kind = exec_kind

        self._on_start = on_start
        self._on_result = on_result
        self._on_finish = on_finish

    # custom request lifecycle
    def unroll_job_list(self) -> list[object]:
        jobs = self.assembler.unroll_job_list(self.request)
        return cast(list[object], jobs)

    def on_start_hook(self) -> None:
        if self._on_start is not None:
            self._on_start(self.request)

    def run_job(self, job: object) -> object:
        typed_job = cast(WorkJob[JobT], job)
        return self.assembler.run_job(typed_job)

    def on_result_hook(self, result: object) -> None:
        typed_res = cast(ResT, result)
        if self._on_result is not None:
            self._on_result(typed_res)

    def on_finish_hook(self, *, cancelled: bool) -> None:
        if self._on_finish is not None:
            self._on_finish(cancelled)

# internal work request contract
# ---------------------

type AnyWorkRequest = _WorkRequestRuntime

# generic de-typed internal runtime contract
class _WorkRequestRuntime(Protocol):
    @property
    def scope_key(self) -> str: ...
    @property
    def request_key(self) -> Hashable: ...
    @property
    def priority(self) -> RequestPriority: ...
    @property
    def request_kind(self) -> RequestKind: ...
    @property
    def exec_kind(self) -> ExecutionKind: ...

    def unroll_job_list(self) -> list[object]:
        ...
    def on_start_hook(self) -> None:
        ...
    def run_job(self, job: object) -> object:
        ...
    def on_result_hook(self, result: object) -> None:
        ...
    def on_finish_hook(self, *, cancelled: bool) -> None:
        ...


@dataclass
class _RequestState:
    request_id:     int
    request:        AnyWorkRequest
    pending_jobs:   int
    cancelled:      bool = False


@dataclass(order=True)
class _QueuedJob:
    priority:       RequestPriority
    seq:            int
    request_id:     int
    request:        AnyWorkRequest = field(compare=False)
    job:            object = field(compare=False)


@dataclass(frozen=True)
class _ResultWrapper:
    request_id:     int
    scope_key:      str
    result:         object | None = None
    error:          BaseException | None = None


@dataclass(frozen=True)
class WorkerPoolConfig:
    thread_workers:     int = 4
    reserved_workers:   int = 2
    process_workers:    int = 0

    @property
    def shared_workers(self) -> int:
        return max(0, self.thread_workers - self.reserved_workers)

# ------------------

class WorkManager:

    def __init__(
        self,
        *,
        schedule_display_callback: DisplayScheduler,
        max_workers: int = 4,
        reserved_workers: int = 2,
        process_workers: int = 0,
        error_handler: ErrorHandler | None = None,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if reserved_workers < 0:
            raise ValueError("reserved_workers must be >= 0")
        if reserved_workers > max_workers:
            raise ValueError("max_workers must be >= reserved_workers")
        if process_workers < 0:
            raise ValueError("process_workers must be >= 0")

        self.pool_config = WorkerPoolConfig(
            thread_workers      = max_workers,
            reserved_workers    = reserved_workers,
            process_workers     = process_workers
        )

        self._reserved_executor = (
            ThreadPoolExecutor(
                max_workers=self.pool_config.reserved_workers
            )
            if self.pool_config.reserved_workers > 0 else None
        )
        self._shared_executor = (
            ThreadPoolExecutor(
                max_workers=self.pool_config.shared_workers
            )
            if self.pool_config.shared_workers > 0 else None
        )
        self._process_executor: ProcessPoolExecutor | None = None

        self._schedule_display_callback = schedule_display_callback
        self._error_handler = error_handler or self.default_error_handler

        self._lock = Lock()
        self._next_req_id = 1
        self._next_job_seq = 1
        self._active_by_scope: dict[str, _RequestState] = {}

        self._pending_ui_thread_jobs: list[_QueuedJob] = []
        self._pending_bg_thread_jobs: list[_QueuedJob] = []
        self._pending_process_jobs: list[_QueuedJob] = []

        self._active_reserved_workers: int = 0
        self._active_shared_workers: int = 0
        self._active_process_workers: int = 0

        self._result_queue: Queue[_ResultWrapper] = Queue()
        self._drain_scheduled = False

    # -------------------------------------------
    # |               Public API                |
    # -------------------------------------------

    def submit(self, request: AnyWorkRequest) -> int:
        jobs = list(request.unroll_job_list())

        cancelled_req: AnyWorkRequest | None = None
        request_id: int

        # check if request still current
        with self._lock:
            current_state = self._active_by_scope.get(request.scope_key)
            if (
                current_state is not None
                and not current_state.cancelled
                and current_state.request.request_key == request.request_key
            ):
                return current_state.request_id

            if current_state is not None and not current_state.cancelled:
                current_state.cancelled = True
                cancelled_req = current_state.request

            request_id = self._next_req_id
            self._next_req_id += 1

            self._active_by_scope[request.scope_key] = _RequestState(
                request_id      = request_id,
                request         = request,
                pending_jobs    = len(jobs),
            )
        if cancelled_req is not None:
            cancelled_req.on_finish_hook(cancelled=True)

        # start processing request
        request.on_start_hook()

        if not jobs:
            # empty request; finish it
            finished_req = self._pop_if_finished(
                scope_key           = request.scope_key,
                request_id          = request_id,
                ignore_pending      = True,
            )
            if finished_req is not None:
                finished_req.on_finish_hook(cancelled=False)
            return request_id

        with self._lock:
            self._enqueue_jobs(request_id, request, jobs)

        # run scheduler tick
        self._dispatch_jobs()
        return request_id

    def cancel_scope(self, scope_key: str) -> None:
        cancelled_req: AnyWorkRequest | None = None

        # check if already cancelled
        with self._lock:
            current_state = self._active_by_scope.pop(scope_key, None)
            if current_state is not None and not current_state.cancelled:
                current_state.cancelled = True
                cancelled_req = current_state.request
        if cancelled_req is not None:
            cancelled_req.on_finish_hook(cancelled=True)

    def shutdown(self, *, wait: bool = False) -> None:
        if self._reserved_executor is not None:
            self._reserved_executor.shutdown(wait=wait, cancel_futures=False)
        if self._shared_executor is not None:
            self._shared_executor.shutdown(wait=wait, cancel_futures=False)
        if self._process_executor is not None:
            self._process_executor.shutdown(wait=wait, cancel_futures=False)

    # -------------------------------------------
    # |               Scheduler                 |
    # -------------------------------------------

    # must hold lock to call
    def _enqueue_jobs(
        self,
        request_id: int,
        request: AnyWorkRequest,
        jobs: list[object],
    ) -> None:
        for job in jobs:
            job_to_queue = _QueuedJob(
                priority    = request.priority,
                seq         = self._next_job_seq,
                request_id  = request_id,
                request     = request,
                job         = job,
            )
            self._next_job_seq += 1

            if request.exec_kind == "thread":
                if request.request_kind == "update":
                    heapq.heappush(
                        self._pending_ui_thread_jobs,
                        job_to_queue
                    )
                else:
                    heapq.heappush(
                        self._pending_bg_thread_jobs,
                        job_to_queue
                    )
            elif request.exec_kind == "process":
                if self._process_executor is None:
                    raise NotImplementedError
                heapq.heappush(
                    self._pending_process_jobs,
                    job_to_queue
                )
            else:
                raise ValueError(f"unsupported exec_kind: {request.exec_kind}")

    def _dispatch_jobs(self) -> None:
        while True:
            # dispatch jobs until all lanes are full
            dispatch = self._next_job()
            if dispatch is None:
                return

            lane, queued = dispatch
            if lane == "reserved":
                assert self._reserved_executor is not None
                self._reserved_executor.submit(
                    self._run_thread_job,
                    lane,
                    queued.request_id,
                    queued.request,
                    queued.job,
                )
            elif lane == "shared":
                assert self._shared_executor is not None
                self._shared_executor.submit(
                    self._run_thread_job,
                    lane,
                    queued.request_id,
                    queued.request,
                    queued.job,
                )
            elif lane == "process":
                raise NotImplementedError
            else:
                raise ValueError(f"unsupported dispatch lane: {lane}")

    def _next_job(self) -> tuple[DispatchLane, _QueuedJob] | None:
        with self._lock:
            next_job = self._next_reserved_job()
            if next_job is not None:
                self._active_reserved_workers += 1
                return ("reserved", next_job)

            next_job = self._next_shared_job()
            if next_job is not None:
                self._active_shared_workers += 1
                return ("shared", next_job)

            next_job = self._next_process_job()
            if next_job is not None:
                self._active_process_workers += 1
                return ("process", next_job)

            return None

    # WARNING: must hold _lock to call!
    def _next_reserved_job(self) -> _QueuedJob | None:
        if self._reserved_executor is None:
            return None
        if self._active_reserved_workers >= self.pool_config.reserved_workers:
            return None
        return self._pop_valid_job(self._pending_ui_thread_jobs)

    # WARNING: must hold _lock to call!
    def _next_shared_job(self) -> _QueuedJob | None:
        if self._shared_executor is None:
            return None
        if self._active_shared_workers >= self.pool_config.shared_workers:
            return None

        queued_job = self._pop_valid_job(self._pending_ui_thread_jobs)
        if queued_job is not None:
            return queued_job

        return self._pop_valid_job(self._pending_bg_thread_jobs) 

    # WARNING: must hold _lock to call!
    def _next_process_job(self) -> _QueuedJob | None:
        if self._process_executor is None:
            return None
        if self._active_process_workers >= self.pool_config.process_workers:
            return None
        return self._pop_valid_job(self._pending_process_jobs)

    # WARNING: must hold _lock to call!
    def _pop_valid_job(self, heap: list[_QueuedJob]) -> _QueuedJob | None:
        while heap:
            queued_job = heapq.heappop(heap)

            # check if still current
            current = self._active_by_scope.get(queued_job.request.scope_key)
            if (
                current is not None
                and current.request_id == queued_job.request_id
                and not current.cancelled
            ):
                return queued_job
        return None

    # -------------------------------------------
    # |             Worker Path                 |
    # -------------------------------------------

    def _run_thread_job(
        self,
        lane: DispatchLane,
        request_id: int,
        request: AnyWorkRequest,
        job: object
    ) -> None:
        try:
            result = request.run_job(job)
            res = _ResultWrapper(
                request_id  = request_id,
                scope_key   = request.scope_key,
                result      = result,
                error       = None,
            )
        except BaseException as exc:
            res = _ResultWrapper(
                request_id  = request_id,
                scope_key   = request.scope_key,
                result      = None,
                error       = exc,
            )

        self._result_queue.put(res)
        self._schedule_drain()

        with self._lock:
            if lane == "reserved":
                self._active_reserved_workers -= 1
            elif lane == "shared":
                self._active_shared_workers -= 1
            else:
                raise ValueError(f"unsupported thread lane: {lane}")

        self._dispatch_jobs()

    # -------------------------------------------
    # |           Display-Thread Path           |
    # -------------------------------------------

    def drain_results(self) -> None:
        # empty result queue
        while True:
            try:
                res_wrapper = self._result_queue.get_nowait()
            except Empty:
                break
            self._unwrap_result(res_wrapper)

        reschedule = False
        with self._lock:
            self._drain_scheduled = False
            if not self._result_queue.empty():
                self._drain_scheduled = True
                reschedule = True
        if reschedule:
            self._schedule_display_callback(self.drain_results)

    def _schedule_drain(self) -> None:
        should_schedule = False
        with self._lock:
            if not self._drain_scheduled:
                self._drain_scheduled = True
                should_schedule = True
        if should_schedule:
            self._schedule_display_callback(self.drain_results)

    def _unwrap_result(self, wrapper: _ResultWrapper) -> None:
        with self._lock:
            current_state = self._active_by_scope.get(wrapper.scope_key)
            if current_state is None:
                return
            if current_state.request_id != wrapper.request_id:
                return
            if current_state.cancelled:
                return
            request = current_state.request

        if wrapper.error is not None:
            self._error_handler(
                wrapper.scope_key,
                wrapper.request_id,
                wrapper.error
            )
        else:
            request.on_result_hook(wrapper.result)

        # try to finish request
        finished_req = self._pop_if_finished(
            scope_key           = wrapper.scope_key,
            request_id          = wrapper.request_id,
            ignore_pending      = False,
        )
        if finished_req is not None:
            finished_req.on_finish_hook(cancelled=False)

    def _pop_if_finished(
        self,
        *,
        scope_key: str,
        request_id: int,
        ignore_pending: bool,
    ) -> AnyWorkRequest | None:
        with self._lock:
            current_state = self._active_by_scope.get(scope_key)
            if current_state is None:
                return None
            if current_state.request_id != request_id:
                return None
            if current_state.cancelled:
                return None

            if not ignore_pending:
                current_state.pending_jobs -= 1
                if current_state.pending_jobs > 0:
                    return None

            finished_req = current_state.request
            self._active_by_scope.pop(scope_key, None)
            return finished_req

    @staticmethod
    def default_error_handler(
        scope_key: str,
        request_id: int,
        exc: BaseException
    ) -> None:
        print(
            f"[WorkManager] scope={scope_key!r} "
            + f"request_id={request_id} "
            + f"error={exc!r}"
        )


