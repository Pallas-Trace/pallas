//
// Created by khatharsis on 23/01/25.
//
#include "pallas_python.h"

std::string Token_toString(pallas::Token t) {
  std::string out;
  switch (t.type) {
  case pallas::TypeInvalid:
    out += "U";
    break;
  case pallas::TypeEvent:
    out += "E";
    break;
  case pallas::TypeSequence:
    out += "S";
    break;
  case pallas::TypeLoop:
    out += "L";
    break;
  }
  out += std::to_string(t.id);
  return out;
}

static inline void pop_data(pallas::Event* e, void* data, size_t data_size, byte*& cursor) {
  if (cursor == nullptr) {
    cursor = &e->event_data[0];
  }
  memcpy(data, cursor, data_size);
  cursor += data_size;
}

#define READ(type, name)                      \
  {                                           \
    type name;                                \
    pop_data(e, &name, sizeof(type), cursor); \
    dict[#name] = name;                       \
  }

static py::dict& Event_get_data(pallas::Event* e) {
  auto& dict = *new py::dict();
  byte* cursor = nullptr;

  switch (e->record) {
  case pallas::PALLAS_EVENT_ENTER:
  case pallas::PALLAS_EVENT_LEAVE: {
    READ(pallas::RegionRef, region_ref);
    break;
  }
  case pallas::PALLAS_EVENT_THREAD_FORK:
  case pallas::PALLAS_EVENT_OMP_FORK: {
    READ(uint32_t, numberOfRequestedThreads);
    break;
  }
  case pallas::PALLAS_EVENT_MPI_SEND:
  case pallas::PALLAS_EVENT_MPI_RECV: {
    READ(uint32_t, receiver);
    READ(uint32_t, communicator);
    READ(uint32_t, msgTag);
    READ(uint64_t, msgLength);
    break;
  }
  case pallas::PALLAS_EVENT_MPI_ISEND:
  case pallas::PALLAS_EVENT_MPI_IRECV: {
    READ(uint32_t, receiver);
    READ(uint32_t, communicator);
    READ(uint32_t, msgTag);
    READ(uint64_t, msgLength);
    READ(uint64_t, requestID);
    break;
  }
  case pallas::PALLAS_EVENT_MPI_ISEND_COMPLETE:
  case pallas::PALLAS_EVENT_MPI_IRECV_REQUEST: {
    READ(uint64_t, requestID);
    break;
  }
  case pallas::PALLAS_EVENT_THREAD_ACQUIRE_LOCK:
  case pallas::PALLAS_EVENT_THREAD_RELEASE_LOCK:
  case pallas::PALLAS_EVENT_OMP_ACQUIRE_LOCK:
  case pallas::PALLAS_EVENT_OMP_RELEASE_LOCK: {
    READ(uint32_t, lockID);
    READ(uint32_t, acquisitionOrder);
    break;
  }
  case pallas::PALLAS_EVENT_MPI_COLLECTIVE_END: {
    READ(uint32_t, collectiveOp);
    READ(uint32_t, communicator);
    READ(uint32_t, root);
    READ(uint64_t, sizeSent);
    READ(uint64_t, sizeReceived);
    break;
  }
  case pallas::PALLAS_EVENT_OMP_TASK_CREATE:
  case pallas::PALLAS_EVENT_OMP_TASK_SWITCH:
  case pallas::PALLAS_EVENT_OMP_TASK_COMPLETE: {
    READ(uint64_t, taskID);
    break;
  }
  case pallas::PALLAS_EVENT_GENERIC: {
    pallas::StringRef event_name;
    pop_data(e, &event_name, sizeof(event_name), cursor);
    dict["event_name"] = event_name;
    break;
  }
  case pallas::PALLAS_EVENT_THREAD_BEGIN:
  case pallas::PALLAS_EVENT_THREAD_END:
  case pallas::PALLAS_EVENT_THREAD_TEAM_BEGIN:
  case pallas::PALLAS_EVENT_THREAD_TEAM_END:
  case pallas::PALLAS_EVENT_THREAD_JOIN:
  case pallas::PALLAS_EVENT_MPI_COLLECTIVE_BEGIN:
  case pallas::PALLAS_EVENT_THREAD_TASK_CREATE:
  case pallas::PALLAS_EVENT_THREAD_TASK_COMPLETE:
  case pallas::PALLAS_EVENT_THREAD_TASK_SWITCH:
    // No additional data for these events
  default:
    break;
  }

  return dict;
}

void setupEnums(const py::module_& m) {
  py::enum_<pallas::TokenType>(m, "TokenType")
    .value("INVALID", pallas::TypeInvalid)
    .value("EVENT", pallas::TypeEvent)
    .value("SEQUENCE", pallas::TypeSequence)
    .value("LOOP", pallas::TypeLoop)
    .export_values();

  py::enum_<pallas::Record>(m, "Record")
    .value("BUFFER_FLUSH", pallas::PALLAS_EVENT_BUFFER_FLUSH)
    .value("MEASUREMENT_ON_OFF", pallas::PALLAS_EVENT_MEASUREMENT_ON_OFF)
    .value("ENTER", pallas::PALLAS_EVENT_ENTER)
    .value("LEAVE", pallas::PALLAS_EVENT_LEAVE)
    .value("MPI_SEND", pallas::PALLAS_EVENT_MPI_SEND)
    .value("MPI_ISEND", pallas::PALLAS_EVENT_MPI_ISEND)
    .value("MPI_ISEND_COMPLETE", pallas::PALLAS_EVENT_MPI_ISEND_COMPLETE)
    .value("MPI_IRECV_REQUEST", pallas::PALLAS_EVENT_MPI_IRECV_REQUEST)
    .value("MPI_RECV", pallas::PALLAS_EVENT_MPI_RECV)
    .value("MPI_IRECV", pallas::PALLAS_EVENT_MPI_IRECV)
    .value("MPI_REQUEST_TEST", pallas::PALLAS_EVENT_MPI_REQUEST_TEST)
    .value("MPI_REQUEST_CANCELLED", pallas::PALLAS_EVENT_MPI_REQUEST_CANCELLED)
    .value("MPI_COLLECTIVE_BEGIN", pallas::PALLAS_EVENT_MPI_COLLECTIVE_BEGIN)
    .value("MPI_COLLECTIVE_END", pallas::PALLAS_EVENT_MPI_COLLECTIVE_END)
    .value("OMP_FORK", pallas::PALLAS_EVENT_OMP_FORK)
    .value("OMP_JOIN", pallas::PALLAS_EVENT_OMP_JOIN)
    .value("OMP_ACQUIRE_LOCK", pallas::PALLAS_EVENT_OMP_ACQUIRE_LOCK)
    .value("OMP_RELEASE_LOCK", pallas::PALLAS_EVENT_OMP_RELEASE_LOCK)
    .value("OMP_TASK_CREATE", pallas::PALLAS_EVENT_OMP_TASK_CREATE)
    .value("OMP_TASK_SWITCH", pallas::PALLAS_EVENT_OMP_TASK_SWITCH)
    .value("OMP_TASK_COMPLETE", pallas::PALLAS_EVENT_OMP_TASK_COMPLETE)
    .value("METRIC", pallas::PALLAS_EVENT_METRIC)
    .value("PARAMETER_STRING", pallas::PALLAS_EVENT_PARAMETER_STRING)
    .value("PARAMETER_INT", pallas::PALLAS_EVENT_PARAMETER_INT)
    .value("PARAMETER_UNSIGNED_INT", pallas::PALLAS_EVENT_PARAMETER_UNSIGNED_INT)
    .value("THREAD_FORK", pallas::PALLAS_EVENT_THREAD_FORK)
    .value("THREAD_JOIN", pallas::PALLAS_EVENT_THREAD_JOIN)
    .value("THREAD_TEAM_BEGIN", pallas::PALLAS_EVENT_THREAD_TEAM_BEGIN)
    .value("THREAD_TEAM_END", pallas::PALLAS_EVENT_THREAD_TEAM_END)
    .value("THREAD_ACQUIRE_LOCK", pallas::PALLAS_EVENT_THREAD_ACQUIRE_LOCK)
    .value("THREAD_RELEASE_LOCK", pallas::PALLAS_EVENT_THREAD_RELEASE_LOCK)
    .value("THREAD_TASK_CREATE", pallas::PALLAS_EVENT_THREAD_TASK_CREATE)
    .value("THREAD_TASK_SWITCH", pallas::PALLAS_EVENT_THREAD_TASK_SWITCH)
    .value("THREAD_TASK_COMPLETE", pallas::PALLAS_EVENT_THREAD_TASK_COMPLETE)
    .value("THREAD_CREATE", pallas::PALLAS_EVENT_THREAD_CREATE)
    .value("THREAD_BEGIN", pallas::PALLAS_EVENT_THREAD_BEGIN)
    .value("THREAD_WAIT", pallas::PALLAS_EVENT_THREAD_WAIT)
    .value("THREAD_END", pallas::PALLAS_EVENT_THREAD_END)
    .value("IO_CREATE_HANDLE", pallas::PALLAS_EVENT_IO_CREATE_HANDLE)
    .value("IO_DESTROY_HANDLE", pallas::PALLAS_EVENT_IO_DESTROY_HANDLE)
    .value("IO_DUPLICATE_HANDLE", pallas::PALLAS_EVENT_IO_DUPLICATE_HANDLE)
    .value("IO_SEEK", pallas::PALLAS_EVENT_IO_SEEK)
    .value("IO_CHANGE_STATUS_FLAGS", pallas::PALLAS_EVENT_IO_CHANGE_STATUS_FLAGS)
    .value("IO_DELETE_FILE", pallas::PALLAS_EVENT_IO_DELETE_FILE)
    .value("IO_OPERATION_BEGIN", pallas::PALLAS_EVENT_IO_OPERATION_BEGIN)
    .value("IO_OPERATION_TEST", pallas::PALLAS_EVENT_IO_OPERATION_TEST)
    .value("IO_OPERATION_ISSUED", pallas::PALLAS_EVENT_IO_OPERATION_ISSUED)
    .value("IO_OPERATION_COMPLETE", pallas::PALLAS_EVENT_IO_OPERATION_COMPLETE)
    .value("IO_OPERATION_CANCELLED", pallas::PALLAS_EVENT_IO_OPERATION_CANCELLED)
    .value("IO_ACQUIRE_LOCK", pallas::PALLAS_EVENT_IO_ACQUIRE_LOCK)
    .value("IO_RELEASE_LOCK", pallas::PALLAS_EVENT_IO_RELEASE_LOCK)
    .value("IO_TRY_LOCK", pallas::PALLAS_EVENT_IO_TRY_LOCK)
    .value("PROGRAM_BEGIN", pallas::PALLAS_EVENT_PROGRAM_BEGIN)
    .value("PROGRAM_END", pallas::PALLAS_EVENT_PROGRAM_END)
    .value("NON_BLOCKING_COLLECTIVE_REQUEST", pallas::PALLAS_EVENT_NON_BLOCKING_COLLECTIVE_REQUEST)
    .value("NON_BLOCKING_COLLECTIVE_COMPLETE", pallas::PALLAS_EVENT_NON_BLOCKING_COLLECTIVE_COMPLETE)
    .value("COMM_CREATE", pallas::PALLAS_EVENT_COMM_CREATE)
    .value("COMM_DESTROY", pallas::PALLAS_EVENT_COMM_DESTROY)
    .value("GENERIC", pallas::PALLAS_EVENT_GENERIC)
    .export_values();
}

std::vector<pallas::Thread*> Archive_get_threads(pallas::Archive& archive) {
  auto vector = std::vector<pallas::Thread*>();
  for (size_t i = 0; i < archive.nb_threads; ++i) {
    vector.push_back(archive.getThreadAt(i));
  }
  return vector;
}

struct PyLocationGroup {
  const pallas::LocationGroup& lg;
  const pallas::GlobalArchive& trace;
};
struct PyLocation {
  pallas::Location& loc;
  const pallas::GlobalArchive& trace;
};
struct PyRegion {
  pallas::Region& reg;
  const pallas::GlobalArchive& trace;
};

std::map<pallas::ThreadId, PyLocation>& Trace_get_locations(pallas::GlobalArchive& trace) {
  auto& map = *new std::map<pallas::ThreadId, PyLocation>();
  for (auto& loc : trace.locations) {
    map.insert(std::pair(loc.id, PyLocation{loc, trace}));
  }
  return map;
}

std::map<pallas::LocationGroupId, PyLocationGroup>& Trace_get_location_groups(pallas::GlobalArchive& trace) {
  auto& map = *new std::map<pallas::LocationGroupId, PyLocationGroup>();
  for (auto& lg : trace.location_groups) {
    map.insert(std::pair(lg.id, PyLocationGroup{lg, trace}));
  }
  return map;
}

std::map<pallas::RegionRef, PyRegion>& Trace_get_regions(pallas::GlobalArchive& trace) {
  auto& map = *new std::map<pallas::RegionRef, PyRegion>();
  for (auto& [key, r] : trace.definitions.regions) {
    map.insert(std::pair(key, PyRegion{r, trace}));
  }
  return map;
}

py::list& Trace_get_archives(pallas::GlobalArchive& trace) {
  auto& list = *new py::list(trace.location_groups.size());
  int i = 0;
  for (auto& locationGroup : trace.location_groups) {
      list[i++] = trace.getArchive(locationGroup.id);
  }
  return list;
}

pallas::GlobalArchive* open_trace(const std::string& path) {
  return pallas_open_trace(path.c_str());
}

class DataHolder {
 private:
  pallas::LinkedVector& data;

 public:
  explicit DataHolder(pallas::LinkedVector& data_) : data(data_) {};
  py::array_t<uint64_t> get_array() {
    return py::array_t({data.size}, {sizeof(uint64_t)}, &data.front(),  //
                       py::capsule(this, [](void* p) {
                         auto* holder = reinterpret_cast<DataHolder*>(p);
                         if (holder->data.size > 3) {
                           holder->data.deleteTimestamps();
                           // TODO Don't delete it, but rather use the LRU
                         }
                         delete holder;
                       }));
  }
};

PYBIND11_MODULE(pallas, m) {
  m.doc() = "Python API for the Pallas library";

  setupEnums(m);

  py::class_<pallas::Token>(m, "Token", "A Pallas token")
    .def_property_readonly("id", [](pallas::Token t) { return t.id; })
    .def_property_readonly("type", [](pallas::Token t) { return t.type; })
    .def("__repr__", &Token_toString);

  py::class_<pallas::Sequence>(m, "Sequence", "A Pallas Sequence, ie a group of tokens.")
    .def_readonly("id", &pallas::Sequence::id)
    .def_readonly("tokens", &pallas::Sequence::tokens)
    .def_property_readonly("timestamps", [](const pallas::Sequence& self) { return (new DataHolder(*self.timestamps))->get_array(); })
    .def_property_readonly("durations", [](const pallas::Sequence& self) { return (new DataHolder(*self.durations))->get_array(); })
    .def_property_readonly("max_duration", [](const pallas::Sequence& self) { return self.durations->max; })
    .def_property_readonly("min_duration", [](const pallas::Sequence& self) { return self.durations->min; })
    .def_property_readonly("mean_duration", [](const pallas::Sequence& self) { return self.durations->mean; })
    .def("guessName", [](pallas::Sequence& self, const pallas::Thread* thread) { return self.guessName(thread); })
    .def("__repr__", [](const pallas::Sequence& self) { return "<pallas_python.Sequence " + std::to_string(self.id) + ">"; });

  py::class_<pallas::Loop>(m, "Loop", "A Pallas Loop, ie a repetition of a Sequence token.")
    .def_readonly("id", &pallas::Loop::self_id)
    .def_readonly("repeated_token", &pallas::Loop::repeated_token)
    .def_readonly("nb_iterations", &pallas::Loop::nb_iterations)
    .def("__repr__", [](const pallas::Loop& self) { return "<pallas_python.Loop " + std::to_string(self.self_id.id) + ">"; });

  py::class_<pallas::EventSummary>(m, "EventSummary", "A Pallas Event Summary, that stores info about an event.")
    .def_readonly("id", &pallas::EventSummary::id)
    .def_readonly("event", &pallas::EventSummary::event)
    .def_readonly("nb_occurences", &pallas::EventSummary::nb_occurences)
    .def_property_readonly("durations", [](const pallas::EventSummary& self) { return (new DataHolder(*self.durations))->get_array(); })
    .def_property_readonly("max_duration", [](const pallas::EventSummary& self) { return self.durations->max; })
    .def_property_readonly("min_duration", [](const pallas::EventSummary& self) { return self.durations->min; })
    .def_property_readonly("mean_duration", [](const pallas::EventSummary& self) { return self.durations->mean; })
    .def("__repr__", [](const pallas::EventSummary& self) { return "<pallas_python.EventSummary " + std::to_string(self.id) + ">"; });

  py::class_<pallas::Event>(m, "Event", "A Pallas Event.").def_readonly("record", &pallas::Event::record).def_property_readonly("data", &Event_get_data);

  py::class_<pallas::Thread>(m, "Thread", "A Pallas thread.")
    .def_readonly("id", &pallas::Thread::id)
    .def_property_readonly("events", [](pallas::Thread& self) { return std::vector(self.events, self.events + self.nb_events); })
    .def_property_readonly("sequences", [](pallas::Thread& self) { return std::vector(self.sequences, self.sequences + self.nb_sequences); })
    .def_property_readonly("loops", [](pallas::Thread& self) { return std::vector(self.loops, self.loops + self.nb_loops); })
    .def("__repr__", [](const pallas::Thread& self) { return "<pallas_python.Thread " + std::to_string(self.id) + ">"; });

  py::class_<PyLocationGroup>(m, "LocationGroup", "A group of Pallas locations. Usually means a process.")
    .def_property_readonly("id", [](const PyLocationGroup& lg) { return lg.lg.id; })
    .def_property_readonly("name", [](const PyLocationGroup& lg) { return lg.trace.definitions.getString(lg.lg.name); })
    .def_property_readonly("parent", [](const PyLocationGroup& lg) { return lg.trace.getLocationGroup(lg.lg.parent); })
    .def_property_readonly("main_location", [](const PyLocationGroup& lg) { return lg.trace.getLocation(lg.lg.mainLoc); })
    .def("__repr__", [](const PyLocationGroup& self) {
      return "<pallas_python.LocationGroup " + std::to_string(self.lg.id) + ": '" + self.trace.definitions.getString(self.lg.name)->str + "'>";
    });
  py::class_<PyLocation>(m, "Location", "A Pallas location. Usually means an execution thread.")
    .def_property_readonly("id", [](const PyLocation& lg) { return lg.loc.id; })
    .def_property_readonly("name", [](const PyLocation& lg) { return lg.trace.definitions.getString(lg.loc.name); })
    .def_property_readonly("parent", [](const PyLocation& lg) { return lg.trace.getLocationGroup(lg.loc.parent); })
    .def("__repr__", [](const PyLocation& self) {
      auto name = self.trace.definitions.getString(self.loc.name);
      return "<pallas_python.Location " + std::to_string(self.loc.id) + ": '" + name->str + "'>";
    });
  py::class_<PyRegion>(m, "Region", "A Pallas region.")
    .def_property_readonly("id", [](const PyRegion& reg) { return reg.reg.region_ref; })
    .def_property_readonly("name", [](const PyRegion& reg) { return reg.trace.definitions.getString(reg.reg.string_ref); })
    .def("__repr__", [](const PyRegion& self) {
      auto name = self.trace.definitions.getString(self.reg.string_ref);
      return "<pallas_python.Region " + std::to_string(self.reg.region_ref) + ": '" + name->str + "'>";
    });

  py::class_<pallas::Archive>(m, "Archive", "A Pallas archive. If it exists, it's already been loaded.")
    .def_readonly("dir_name", &pallas::Archive::dir_name)
    .def_readonly("trace_name", &pallas::Archive::trace_name)
    .def_property_readonly("threads", &Archive_get_threads);

  m.def("open_trace", &open_trace, "Open a Pallas trace");

  py::class_<pallas::GlobalArchive>(m, "Trace", "A Pallas Trace file.")
    .def(py::init(&open_trace), "Open a trace file and read its structure.")
    .def_readonly("dir_name", &pallas::GlobalArchive::dir_name)
    .def_readonly("trace_name", &pallas::GlobalArchive::trace_name)
    .def_readonly("fullpath", &pallas::GlobalArchive::fullpath)
    .def_property_readonly("locations", &Trace_get_locations)
    .def_property_readonly("location_groups", &Trace_get_location_groups)
    .def_property_readonly("strings", [](pallas::GlobalArchive& trace) { return trace.definitions.strings; })
    .def_property_readonly("regions", &Trace_get_regions)
    .def_property_readonly("archives", &Trace_get_archives);
}