"""
Python API for the Pallas library
"""
from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing
__all__: list[str] = ['Archive', 'BUFFER_FLUSH', 'COMM_CREATE', 'COMM_DESTROY', 'ENTER', 'EVENT', 'Event', 'GENERIC', 'INVALID', 'IO_ACQUIRE_LOCK', 'IO_CHANGE_STATUS_FLAGS', 'IO_CREATE_HANDLE', 'IO_DELETE_FILE', 'IO_DESTROY_HANDLE', 'IO_DUPLICATE_HANDLE', 'IO_OPERATION_BEGIN', 'IO_OPERATION_CANCELLED', 'IO_OPERATION_COMPLETE', 'IO_OPERATION_ISSUED', 'IO_OPERATION_TEST', 'IO_RELEASE_LOCK', 'IO_SEEK', 'IO_TRY_LOCK', 'LEAVE', 'LOOP', 'Location', 'LocationGroup', 'Loop', 'MEASUREMENT_ON_OFF', 'METRIC', 'MPI_COLLECTIVE_BEGIN', 'MPI_COLLECTIVE_END', 'MPI_IRECV', 'MPI_IRECV_REQUEST', 'MPI_ISEND', 'MPI_ISEND_COMPLETE', 'MPI_RECV', 'MPI_REQUEST_CANCELLED', 'MPI_REQUEST_TEST', 'MPI_SEND', 'NON_BLOCKING_COLLECTIVE_COMPLETE', 'NON_BLOCKING_COLLECTIVE_REQUEST', 'OMP_ACQUIRE_LOCK', 'OMP_FORK', 'OMP_JOIN', 'OMP_RELEASE_LOCK', 'OMP_TASK_COMPLETE', 'OMP_TASK_CREATE', 'OMP_TASK_SWITCH', 'PARAMETER_INT', 'PARAMETER_STRING', 'PARAMETER_UNSIGNED_INT', 'PROGRAM_BEGIN', 'PROGRAM_END', 'Record', 'Region', 'SEQUENCE', 'Sequence', 'THREAD_ACQUIRE_LOCK', 'THREAD_BEGIN', 'THREAD_CREATE', 'THREAD_END', 'THREAD_FORK', 'THREAD_JOIN', 'THREAD_RELEASE_LOCK', 'THREAD_TASK_COMPLETE', 'THREAD_TASK_CREATE', 'THREAD_TASK_SWITCH', 'THREAD_TEAM_BEGIN', 'THREAD_TEAM_END', 'THREAD_WAIT', 'Thread', 'Thread_Iterator', 'Token', 'TokenType', 'Trace', 'Vector', 'get_ABI', 'open_trace']
class Archive:
    """
    A Pallas archive. If it exists, it's already been loaded.
    """
    @property
    def dir_name(self) -> str:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def locations(self) -> dict[int, Location]:
        ...
    @property
    def metadata(self) -> dict[str, str]:
        ...
    @property
    def regions(self) -> dict[int, Region]:
        ...
    @property
    def strings(self) -> dict[int, str]:
        ...
    @property
    def threads(self) -> list[Thread]:
        ...
class Event:
    """
    A Pallas Event.
    """
    def __repr__(self) -> str:
        ...
    def guessName(self) -> str:
        ...
    @property
    def data(self) -> dict:
        ...
    @property
    def id(self) -> Token:
        ...
    @property
    def nb_occurrences(self) -> int:
        ...
    @property
    def record(self) -> Record:
        ...
    @property
    def timestamps(self) -> Vector:
        ...
class Location:
    """
    A Pallas location. Usually means an execution thread.
    """
    def __repr__(self) -> str:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def parent(self) -> ...:
        ...
class LocationGroup:
    """
    A group of Pallas locations. Usually means a process.
    """
    def __repr__(self) -> str:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def parent(self) -> ...:
        ...
class Loop:
    """
    A Pallas Loop, ie a repetition of a Sequence token.
    """
    def __repr__(self) -> str:
        ...
    @property
    def id(self) -> Token:
        ...
    @property
    def nb_iterations(self) -> int:
        ...
    @property
    def sequence(self) -> typing.Sequence:
        ...
class Record:
    """
    Members:
    
      BUFFER_FLUSH
    
      MEASUREMENT_ON_OFF
    
      ENTER
    
      LEAVE
    
      MPI_SEND
    
      MPI_ISEND
    
      MPI_ISEND_COMPLETE
    
      MPI_IRECV_REQUEST
    
      MPI_RECV
    
      MPI_IRECV
    
      MPI_REQUEST_TEST
    
      MPI_REQUEST_CANCELLED
    
      MPI_COLLECTIVE_BEGIN
    
      MPI_COLLECTIVE_END
    
      OMP_FORK
    
      OMP_JOIN
    
      OMP_ACQUIRE_LOCK
    
      OMP_RELEASE_LOCK
    
      OMP_TASK_CREATE
    
      OMP_TASK_SWITCH
    
      OMP_TASK_COMPLETE
    
      METRIC
    
      PARAMETER_STRING
    
      PARAMETER_INT
    
      PARAMETER_UNSIGNED_INT
    
      THREAD_FORK
    
      THREAD_JOIN
    
      THREAD_TEAM_BEGIN
    
      THREAD_TEAM_END
    
      THREAD_ACQUIRE_LOCK
    
      THREAD_RELEASE_LOCK
    
      THREAD_TASK_CREATE
    
      THREAD_TASK_SWITCH
    
      THREAD_TASK_COMPLETE
    
      THREAD_CREATE
    
      THREAD_BEGIN
    
      THREAD_WAIT
    
      THREAD_END
    
      IO_CREATE_HANDLE
    
      IO_DESTROY_HANDLE
    
      IO_DUPLICATE_HANDLE
    
      IO_SEEK
    
      IO_CHANGE_STATUS_FLAGS
    
      IO_DELETE_FILE
    
      IO_OPERATION_BEGIN
    
      IO_OPERATION_TEST
    
      IO_OPERATION_ISSUED
    
      IO_OPERATION_COMPLETE
    
      IO_OPERATION_CANCELLED
    
      IO_ACQUIRE_LOCK
    
      IO_RELEASE_LOCK
    
      IO_TRY_LOCK
    
      PROGRAM_BEGIN
    
      PROGRAM_END
    
      NON_BLOCKING_COLLECTIVE_REQUEST
    
      NON_BLOCKING_COLLECTIVE_COMPLETE
    
      COMM_CREATE
    
      COMM_DESTROY
    
      GENERIC
    """
    BUFFER_FLUSH: typing.ClassVar[Record]  # value = <Record.BUFFER_FLUSH: 0>
    COMM_CREATE: typing.ClassVar[Record]  # value = <Record.COMM_CREATE: 56>
    COMM_DESTROY: typing.ClassVar[Record]  # value = <Record.COMM_DESTROY: 57>
    ENTER: typing.ClassVar[Record]  # value = <Record.ENTER: 2>
    GENERIC: typing.ClassVar[Record]  # value = <Record.GENERIC: 58>
    IO_ACQUIRE_LOCK: typing.ClassVar[Record]  # value = <Record.IO_ACQUIRE_LOCK: 49>
    IO_CHANGE_STATUS_FLAGS: typing.ClassVar[Record]  # value = <Record.IO_CHANGE_STATUS_FLAGS: 42>
    IO_CREATE_HANDLE: typing.ClassVar[Record]  # value = <Record.IO_CREATE_HANDLE: 38>
    IO_DELETE_FILE: typing.ClassVar[Record]  # value = <Record.IO_DELETE_FILE: 43>
    IO_DESTROY_HANDLE: typing.ClassVar[Record]  # value = <Record.IO_DESTROY_HANDLE: 39>
    IO_DUPLICATE_HANDLE: typing.ClassVar[Record]  # value = <Record.IO_DUPLICATE_HANDLE: 40>
    IO_OPERATION_BEGIN: typing.ClassVar[Record]  # value = <Record.IO_OPERATION_BEGIN: 44>
    IO_OPERATION_CANCELLED: typing.ClassVar[Record]  # value = <Record.IO_OPERATION_CANCELLED: 48>
    IO_OPERATION_COMPLETE: typing.ClassVar[Record]  # value = <Record.IO_OPERATION_COMPLETE: 47>
    IO_OPERATION_ISSUED: typing.ClassVar[Record]  # value = <Record.IO_OPERATION_ISSUED: 46>
    IO_OPERATION_TEST: typing.ClassVar[Record]  # value = <Record.IO_OPERATION_TEST: 45>
    IO_RELEASE_LOCK: typing.ClassVar[Record]  # value = <Record.IO_RELEASE_LOCK: 50>
    IO_SEEK: typing.ClassVar[Record]  # value = <Record.IO_SEEK: 41>
    IO_TRY_LOCK: typing.ClassVar[Record]  # value = <Record.IO_TRY_LOCK: 51>
    LEAVE: typing.ClassVar[Record]  # value = <Record.LEAVE: 3>
    MEASUREMENT_ON_OFF: typing.ClassVar[Record]  # value = <Record.MEASUREMENT_ON_OFF: 1>
    METRIC: typing.ClassVar[Record]  # value = <Record.METRIC: 21>
    MPI_COLLECTIVE_BEGIN: typing.ClassVar[Record]  # value = <Record.MPI_COLLECTIVE_BEGIN: 12>
    MPI_COLLECTIVE_END: typing.ClassVar[Record]  # value = <Record.MPI_COLLECTIVE_END: 13>
    MPI_IRECV: typing.ClassVar[Record]  # value = <Record.MPI_IRECV: 9>
    MPI_IRECV_REQUEST: typing.ClassVar[Record]  # value = <Record.MPI_IRECV_REQUEST: 7>
    MPI_ISEND: typing.ClassVar[Record]  # value = <Record.MPI_ISEND: 5>
    MPI_ISEND_COMPLETE: typing.ClassVar[Record]  # value = <Record.MPI_ISEND_COMPLETE: 6>
    MPI_RECV: typing.ClassVar[Record]  # value = <Record.MPI_RECV: 8>
    MPI_REQUEST_CANCELLED: typing.ClassVar[Record]  # value = <Record.MPI_REQUEST_CANCELLED: 11>
    MPI_REQUEST_TEST: typing.ClassVar[Record]  # value = <Record.MPI_REQUEST_TEST: 10>
    MPI_SEND: typing.ClassVar[Record]  # value = <Record.MPI_SEND: 4>
    NON_BLOCKING_COLLECTIVE_COMPLETE: typing.ClassVar[Record]  # value = <Record.NON_BLOCKING_COLLECTIVE_COMPLETE: 55>
    NON_BLOCKING_COLLECTIVE_REQUEST: typing.ClassVar[Record]  # value = <Record.NON_BLOCKING_COLLECTIVE_REQUEST: 54>
    OMP_ACQUIRE_LOCK: typing.ClassVar[Record]  # value = <Record.OMP_ACQUIRE_LOCK: 16>
    OMP_FORK: typing.ClassVar[Record]  # value = <Record.OMP_FORK: 14>
    OMP_JOIN: typing.ClassVar[Record]  # value = <Record.OMP_JOIN: 15>
    OMP_RELEASE_LOCK: typing.ClassVar[Record]  # value = <Record.OMP_RELEASE_LOCK: 17>
    OMP_TASK_COMPLETE: typing.ClassVar[Record]  # value = <Record.OMP_TASK_COMPLETE: 20>
    OMP_TASK_CREATE: typing.ClassVar[Record]  # value = <Record.OMP_TASK_CREATE: 18>
    OMP_TASK_SWITCH: typing.ClassVar[Record]  # value = <Record.OMP_TASK_SWITCH: 19>
    PARAMETER_INT: typing.ClassVar[Record]  # value = <Record.PARAMETER_INT: 23>
    PARAMETER_STRING: typing.ClassVar[Record]  # value = <Record.PARAMETER_STRING: 22>
    PARAMETER_UNSIGNED_INT: typing.ClassVar[Record]  # value = <Record.PARAMETER_UNSIGNED_INT: 24>
    PROGRAM_BEGIN: typing.ClassVar[Record]  # value = <Record.PROGRAM_BEGIN: 52>
    PROGRAM_END: typing.ClassVar[Record]  # value = <Record.PROGRAM_END: 53>
    THREAD_ACQUIRE_LOCK: typing.ClassVar[Record]  # value = <Record.THREAD_ACQUIRE_LOCK: 29>
    THREAD_BEGIN: typing.ClassVar[Record]  # value = <Record.THREAD_BEGIN: 35>
    THREAD_CREATE: typing.ClassVar[Record]  # value = <Record.THREAD_CREATE: 34>
    THREAD_END: typing.ClassVar[Record]  # value = <Record.THREAD_END: 37>
    THREAD_FORK: typing.ClassVar[Record]  # value = <Record.THREAD_FORK: 25>
    THREAD_JOIN: typing.ClassVar[Record]  # value = <Record.THREAD_JOIN: 26>
    THREAD_RELEASE_LOCK: typing.ClassVar[Record]  # value = <Record.THREAD_RELEASE_LOCK: 30>
    THREAD_TASK_COMPLETE: typing.ClassVar[Record]  # value = <Record.THREAD_TASK_COMPLETE: 33>
    THREAD_TASK_CREATE: typing.ClassVar[Record]  # value = <Record.THREAD_TASK_CREATE: 31>
    THREAD_TASK_SWITCH: typing.ClassVar[Record]  # value = <Record.THREAD_TASK_SWITCH: 32>
    THREAD_TEAM_BEGIN: typing.ClassVar[Record]  # value = <Record.THREAD_TEAM_BEGIN: 27>
    THREAD_TEAM_END: typing.ClassVar[Record]  # value = <Record.THREAD_TEAM_END: 28>
    THREAD_WAIT: typing.ClassVar[Record]  # value = <Record.THREAD_WAIT: 36>
    __members__: typing.ClassVar[dict[str, Record]]  # value = {'BUFFER_FLUSH': <Record.BUFFER_FLUSH: 0>, 'MEASUREMENT_ON_OFF': <Record.MEASUREMENT_ON_OFF: 1>, 'ENTER': <Record.ENTER: 2>, 'LEAVE': <Record.LEAVE: 3>, 'MPI_SEND': <Record.MPI_SEND: 4>, 'MPI_ISEND': <Record.MPI_ISEND: 5>, 'MPI_ISEND_COMPLETE': <Record.MPI_ISEND_COMPLETE: 6>, 'MPI_IRECV_REQUEST': <Record.MPI_IRECV_REQUEST: 7>, 'MPI_RECV': <Record.MPI_RECV: 8>, 'MPI_IRECV': <Record.MPI_IRECV: 9>, 'MPI_REQUEST_TEST': <Record.MPI_REQUEST_TEST: 10>, 'MPI_REQUEST_CANCELLED': <Record.MPI_REQUEST_CANCELLED: 11>, 'MPI_COLLECTIVE_BEGIN': <Record.MPI_COLLECTIVE_BEGIN: 12>, 'MPI_COLLECTIVE_END': <Record.MPI_COLLECTIVE_END: 13>, 'OMP_FORK': <Record.OMP_FORK: 14>, 'OMP_JOIN': <Record.OMP_JOIN: 15>, 'OMP_ACQUIRE_LOCK': <Record.OMP_ACQUIRE_LOCK: 16>, 'OMP_RELEASE_LOCK': <Record.OMP_RELEASE_LOCK: 17>, 'OMP_TASK_CREATE': <Record.OMP_TASK_CREATE: 18>, 'OMP_TASK_SWITCH': <Record.OMP_TASK_SWITCH: 19>, 'OMP_TASK_COMPLETE': <Record.OMP_TASK_COMPLETE: 20>, 'METRIC': <Record.METRIC: 21>, 'PARAMETER_STRING': <Record.PARAMETER_STRING: 22>, 'PARAMETER_INT': <Record.PARAMETER_INT: 23>, 'PARAMETER_UNSIGNED_INT': <Record.PARAMETER_UNSIGNED_INT: 24>, 'THREAD_FORK': <Record.THREAD_FORK: 25>, 'THREAD_JOIN': <Record.THREAD_JOIN: 26>, 'THREAD_TEAM_BEGIN': <Record.THREAD_TEAM_BEGIN: 27>, 'THREAD_TEAM_END': <Record.THREAD_TEAM_END: 28>, 'THREAD_ACQUIRE_LOCK': <Record.THREAD_ACQUIRE_LOCK: 29>, 'THREAD_RELEASE_LOCK': <Record.THREAD_RELEASE_LOCK: 30>, 'THREAD_TASK_CREATE': <Record.THREAD_TASK_CREATE: 31>, 'THREAD_TASK_SWITCH': <Record.THREAD_TASK_SWITCH: 32>, 'THREAD_TASK_COMPLETE': <Record.THREAD_TASK_COMPLETE: 33>, 'THREAD_CREATE': <Record.THREAD_CREATE: 34>, 'THREAD_BEGIN': <Record.THREAD_BEGIN: 35>, 'THREAD_WAIT': <Record.THREAD_WAIT: 36>, 'THREAD_END': <Record.THREAD_END: 37>, 'IO_CREATE_HANDLE': <Record.IO_CREATE_HANDLE: 38>, 'IO_DESTROY_HANDLE': <Record.IO_DESTROY_HANDLE: 39>, 'IO_DUPLICATE_HANDLE': <Record.IO_DUPLICATE_HANDLE: 40>, 'IO_SEEK': <Record.IO_SEEK: 41>, 'IO_CHANGE_STATUS_FLAGS': <Record.IO_CHANGE_STATUS_FLAGS: 42>, 'IO_DELETE_FILE': <Record.IO_DELETE_FILE: 43>, 'IO_OPERATION_BEGIN': <Record.IO_OPERATION_BEGIN: 44>, 'IO_OPERATION_TEST': <Record.IO_OPERATION_TEST: 45>, 'IO_OPERATION_ISSUED': <Record.IO_OPERATION_ISSUED: 46>, 'IO_OPERATION_COMPLETE': <Record.IO_OPERATION_COMPLETE: 47>, 'IO_OPERATION_CANCELLED': <Record.IO_OPERATION_CANCELLED: 48>, 'IO_ACQUIRE_LOCK': <Record.IO_ACQUIRE_LOCK: 49>, 'IO_RELEASE_LOCK': <Record.IO_RELEASE_LOCK: 50>, 'IO_TRY_LOCK': <Record.IO_TRY_LOCK: 51>, 'PROGRAM_BEGIN': <Record.PROGRAM_BEGIN: 52>, 'PROGRAM_END': <Record.PROGRAM_END: 53>, 'NON_BLOCKING_COLLECTIVE_REQUEST': <Record.NON_BLOCKING_COLLECTIVE_REQUEST: 54>, 'NON_BLOCKING_COLLECTIVE_COMPLETE': <Record.NON_BLOCKING_COLLECTIVE_COMPLETE: 55>, 'COMM_CREATE': <Record.COMM_CREATE: 56>, 'COMM_DESTROY': <Record.COMM_DESTROY: 57>, 'GENERIC': <Record.GENERIC: 58>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Region:
    """
    A Pallas region.
    """
    def __repr__(self) -> str:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def name(self) -> str:
        ...
class Sequence:
    """
    A Pallas Sequence, ie a group of tokens.
    """
    def __repr__(self: typing.Sequence) -> str:
        ...
    @typing.overload
    def contains(self: typing.Sequence, arg0: typing.Sequence) -> bool:
        ...
    @typing.overload
    def contains(self: typing.Sequence, arg0: PyLoop) -> bool:
        ...
    @typing.overload
    def contains(self: typing.Sequence, arg0: PyEvent) -> bool:
        ...
    @typing.overload
    def contains(self: typing.Sequence, arg0: Token) -> bool:
        ...
    def guessName(self: typing.Sequence) -> str:
        ...
    @property
    def content(self) -> list:
        ...
    @property
    def durations(self) -> Vector:
        ...
    @property
    def exclusive_durations(self) -> Vector:
        ...
    @property
    def id(self) -> Token:
        ...
    @property
    def max_duration(self) -> int:
        ...
    @property
    def mean_duration(self) -> int:
        ...
    @property
    def min_duration(self) -> int:
        ...
    @property
    def n_iterations(self) -> int:
        ...
    @property
    def timestamps(self) -> Vector:
        ...
    @property
    def tokens(self) -> list[Token]:
        ...
class Thread:
    """
    A Pallas thread.
    """
    def __iter__(self) -> ...:
        ...
    def __repr__(self) -> str:
        ...
    def getSnapshotView(self, arg0: typing.SupportsInt, arg1: typing.SupportsInt) -> dict[Token, int]:
        ...
    def getSnapshotViewFast(self, arg0: typing.SupportsInt, arg1: typing.SupportsInt) -> dict[Token, int]:
        ...
    @typing.overload
    def get_events_from_record(self, arg0: Record) -> list[Event]:
        ...
    @typing.overload
    def get_events_from_record(self, arg0: collections.abc.Sequence[Record]) -> list[Event]:
        ...
    @property
    def events(self) -> list[Event]:
        ...
    @property
    def finish_timestamp(self) -> int:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def loops(self) -> list[Loop]:
        ...
    @property
    def sequences(self) -> list[typing.Sequence]:
        ...
    @property
    def starting_timestamp(self) -> int:
        ...
class Thread_Iterator:
    """
    An iterator over the thread.
    """
    def __next__(self) -> tuple:
        ...
class Token:
    """
    A Pallas token
    """
    def __eq__(self, arg0: Token) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def id(self) -> int:
        ...
    @property
    def type(self) -> TokenType:
        ...
class TokenType:
    """
    Members:
    
      INVALID
    
      EVENT
    
      SEQUENCE
    
      LOOP
    """
    EVENT: typing.ClassVar[TokenType]  # value = <TokenType.EVENT: 1>
    INVALID: typing.ClassVar[TokenType]  # value = <TokenType.INVALID: 0>
    LOOP: typing.ClassVar[TokenType]  # value = <TokenType.LOOP: 3>
    SEQUENCE: typing.ClassVar[TokenType]  # value = <TokenType.SEQUENCE: 2>
    __members__: typing.ClassVar[dict[str, TokenType]]  # value = {'INVALID': <TokenType.INVALID: 0>, 'EVENT': <TokenType.EVENT: 1>, 'SEQUENCE': <TokenType.SEQUENCE: 2>, 'LOOP': <TokenType.LOOP: 3>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Trace:
    """
    A Pallas Trace file.
    """
    def __init__(self, arg0: str) -> None:
        """
        Open a trace file and read its structure.
        """
    @property
    def archives(self) -> list:
        ...
    @property
    def dir_name(self) -> str:
        ...
    @property
    def fullpath(self) -> str:
        ...
    @property
    def location_groups(self) -> dict[int, LocationGroup]:
        ...
    @property
    def locations(self) -> dict[int, Location]:
        ...
    @property
    def metadata(self) -> dict[str, str]:
        ...
    @property
    def regions(self) -> dict[int, Region]:
        ...
    @property
    def strings(self) -> dict[int, str]:
        ...
    @property
    def trace_name(self) -> str:
        ...
class Vector:
    """
    A Pallas custom vector
    """
    def __getitem__(self, arg0: typing.SupportsInt) -> int:
        ...
    def as_numpy_array(self) -> numpy.typing.NDArray[numpy.uint64]:
        ...
    @property
    def size(self) -> int:
        ...
def get_ABI() -> int:
    ...
def open_trace(arg0: str) -> ...:
    """
    Open a Pallas trace
    """
BUFFER_FLUSH: Record  # value = <Record.BUFFER_FLUSH: 0>
COMM_CREATE: Record  # value = <Record.COMM_CREATE: 56>
COMM_DESTROY: Record  # value = <Record.COMM_DESTROY: 57>
ENTER: Record  # value = <Record.ENTER: 2>
EVENT: TokenType  # value = <TokenType.EVENT: 1>
GENERIC: Record  # value = <Record.GENERIC: 58>
INVALID: TokenType  # value = <TokenType.INVALID: 0>
IO_ACQUIRE_LOCK: Record  # value = <Record.IO_ACQUIRE_LOCK: 49>
IO_CHANGE_STATUS_FLAGS: Record  # value = <Record.IO_CHANGE_STATUS_FLAGS: 42>
IO_CREATE_HANDLE: Record  # value = <Record.IO_CREATE_HANDLE: 38>
IO_DELETE_FILE: Record  # value = <Record.IO_DELETE_FILE: 43>
IO_DESTROY_HANDLE: Record  # value = <Record.IO_DESTROY_HANDLE: 39>
IO_DUPLICATE_HANDLE: Record  # value = <Record.IO_DUPLICATE_HANDLE: 40>
IO_OPERATION_BEGIN: Record  # value = <Record.IO_OPERATION_BEGIN: 44>
IO_OPERATION_CANCELLED: Record  # value = <Record.IO_OPERATION_CANCELLED: 48>
IO_OPERATION_COMPLETE: Record  # value = <Record.IO_OPERATION_COMPLETE: 47>
IO_OPERATION_ISSUED: Record  # value = <Record.IO_OPERATION_ISSUED: 46>
IO_OPERATION_TEST: Record  # value = <Record.IO_OPERATION_TEST: 45>
IO_RELEASE_LOCK: Record  # value = <Record.IO_RELEASE_LOCK: 50>
IO_SEEK: Record  # value = <Record.IO_SEEK: 41>
IO_TRY_LOCK: Record  # value = <Record.IO_TRY_LOCK: 51>
LEAVE: Record  # value = <Record.LEAVE: 3>
LOOP: TokenType  # value = <TokenType.LOOP: 3>
MEASUREMENT_ON_OFF: Record  # value = <Record.MEASUREMENT_ON_OFF: 1>
METRIC: Record  # value = <Record.METRIC: 21>
MPI_COLLECTIVE_BEGIN: Record  # value = <Record.MPI_COLLECTIVE_BEGIN: 12>
MPI_COLLECTIVE_END: Record  # value = <Record.MPI_COLLECTIVE_END: 13>
MPI_IRECV: Record  # value = <Record.MPI_IRECV: 9>
MPI_IRECV_REQUEST: Record  # value = <Record.MPI_IRECV_REQUEST: 7>
MPI_ISEND: Record  # value = <Record.MPI_ISEND: 5>
MPI_ISEND_COMPLETE: Record  # value = <Record.MPI_ISEND_COMPLETE: 6>
MPI_RECV: Record  # value = <Record.MPI_RECV: 8>
MPI_REQUEST_CANCELLED: Record  # value = <Record.MPI_REQUEST_CANCELLED: 11>
MPI_REQUEST_TEST: Record  # value = <Record.MPI_REQUEST_TEST: 10>
MPI_SEND: Record  # value = <Record.MPI_SEND: 4>
NON_BLOCKING_COLLECTIVE_COMPLETE: Record  # value = <Record.NON_BLOCKING_COLLECTIVE_COMPLETE: 55>
NON_BLOCKING_COLLECTIVE_REQUEST: Record  # value = <Record.NON_BLOCKING_COLLECTIVE_REQUEST: 54>
OMP_ACQUIRE_LOCK: Record  # value = <Record.OMP_ACQUIRE_LOCK: 16>
OMP_FORK: Record  # value = <Record.OMP_FORK: 14>
OMP_JOIN: Record  # value = <Record.OMP_JOIN: 15>
OMP_RELEASE_LOCK: Record  # value = <Record.OMP_RELEASE_LOCK: 17>
OMP_TASK_COMPLETE: Record  # value = <Record.OMP_TASK_COMPLETE: 20>
OMP_TASK_CREATE: Record  # value = <Record.OMP_TASK_CREATE: 18>
OMP_TASK_SWITCH: Record  # value = <Record.OMP_TASK_SWITCH: 19>
PARAMETER_INT: Record  # value = <Record.PARAMETER_INT: 23>
PARAMETER_STRING: Record  # value = <Record.PARAMETER_STRING: 22>
PARAMETER_UNSIGNED_INT: Record  # value = <Record.PARAMETER_UNSIGNED_INT: 24>
PROGRAM_BEGIN: Record  # value = <Record.PROGRAM_BEGIN: 52>
PROGRAM_END: Record  # value = <Record.PROGRAM_END: 53>
SEQUENCE: TokenType  # value = <TokenType.SEQUENCE: 2>
THREAD_ACQUIRE_LOCK: Record  # value = <Record.THREAD_ACQUIRE_LOCK: 29>
THREAD_BEGIN: Record  # value = <Record.THREAD_BEGIN: 35>
THREAD_CREATE: Record  # value = <Record.THREAD_CREATE: 34>
THREAD_END: Record  # value = <Record.THREAD_END: 37>
THREAD_FORK: Record  # value = <Record.THREAD_FORK: 25>
THREAD_JOIN: Record  # value = <Record.THREAD_JOIN: 26>
THREAD_RELEASE_LOCK: Record  # value = <Record.THREAD_RELEASE_LOCK: 30>
THREAD_TASK_COMPLETE: Record  # value = <Record.THREAD_TASK_COMPLETE: 33>
THREAD_TASK_CREATE: Record  # value = <Record.THREAD_TASK_CREATE: 31>
THREAD_TASK_SWITCH: Record  # value = <Record.THREAD_TASK_SWITCH: 32>
THREAD_TEAM_BEGIN: Record  # value = <Record.THREAD_TEAM_BEGIN: 27>
THREAD_TEAM_END: Record  # value = <Record.THREAD_TEAM_END: 28>
THREAD_WAIT: Record  # value = <Record.THREAD_WAIT: 36>
