#include "python_read.h"

#include <pallas/utils/pallas_storage.h>
#include <pybind11/pytypes.h>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <variant>
#include <vector>
#include <algorithm>
#include <limits>
#include <map>
#include "pallas.h"
#include "pallas_archive.h"
#include "pallas_attribute.h"
#include "pallas_read.h"
#include "utils/pallas_log.h"
#include "utils/pallas_storage.h"

std::vector<pallas::Thread*> Archive_get_threads(pallas::Archive& archive) {
    auto vector = std::vector<pallas::Thread*>();
    for (size_t i = 0; i < archive.nb_threads; ++i) {
        vector.push_back(archive.getThreadAt(i));
    }
    return vector;
}

std::map<pallas::StringRef, std::string> Archive_get_strings(pallas::Archive& archive) {
    auto map = std::map<pallas::StringRef, std::string>();
    for (auto& [key, r] : archive.definitions.strings) {
        map.insert(std::pair(key, archive.getString(r.string_ref)->str));
    }
    return map;
}

std::map<pallas::ThreadId, PyLocation> Archive_get_locations(pallas::Archive& archive) {
    auto map = std::map<pallas::ThreadId, PyLocation>();
    for (auto& loc : archive.locations) {
        map.insert(std::pair(loc.id, PyLocation{loc.id, archive.getString(loc.name)->str, archive.getLocationGroup(loc.parent)}));
    }
    return map;
}

std::map<pallas::RegionRef, PyRegion> Archive_get_regions(pallas::Archive& archive) {
    auto map = std::map<pallas::RegionRef, PyRegion>();
    for (auto& [key, r] : archive.definitions.regions) {
        map.insert(std::pair(key, PyRegion{r.region_ref, archive.getString(r.string_ref)->str}));
    }
    return map;
}

std::map<pallas::ThreadId, PyLocation> Trace_get_locations(pallas::GlobalArchive& trace) {
    auto map = std::map<pallas::ThreadId, PyLocation>();
    for (auto& lg : trace.location_groups) {
        auto* a = trace.getArchive(lg.id);
        for (const auto& [key, value] : Archive_get_locations(*a)) {
            map.insert(std::pair(key, value));
        }
    }
    return map;
}

std::map<pallas::LocationGroupId, PyLocationGroup> Trace_get_location_groups(pallas::GlobalArchive& trace) {
    auto map = std::map<pallas::LocationGroupId, PyLocationGroup>();
    for (auto& lg : trace.location_groups) {
        map.insert(std::pair(lg.id, PyLocationGroup{lg.id, trace.getString(lg.name)->str, trace.getLocationGroup(lg.parent)}));
    }
    return map;
}

std::map<pallas::StringRef, std::string> Trace_get_strings(pallas::GlobalArchive& trace) {
    auto map = std::map<pallas::StringRef, std::string>();
    for (auto& [key, r] : trace.definitions.strings) {
        map.insert(std::pair(key, trace.getString(r.string_ref)->str));
    }
    return map;
}

std::map<pallas::RegionRef, PyRegion> Trace_get_regions(pallas::GlobalArchive& trace) {
    auto map = std::map<pallas::RegionRef, PyRegion>();
    for (auto& [key, r] : trace.definitions.regions) {
        map.insert(std::pair(key, PyRegion{r.region_ref, trace.getString(r.string_ref)->str}));
    }
    return map;
}

namespace {

template <typename A>
std::string string_or_empty(A& archive, pallas::StringRef ref) {
    const auto* s = archive.getString(ref);
    return s ? std::string(s->str) : std::string();
}

template <typename A>
PyMetricMember make_metric_member(A& archive, const pallas::MetricMember& m) {
    return PyMetricMember{m.metric_member_ref,
                          string_or_empty(archive, m.name),
                          string_or_empty(archive, m.description),
                          string_or_empty(archive, m.unit_ref),
                          m.metric_type,
                          m.metric_mode,
                          m.base,
                          m.exponent};
}

PyMetric make_metric(const pallas::Metric& m) {
    return PyMetric{m.metric_ref, std::vector<pallas::MetricMemberRef>(m.metric_members, m.metric_members + m.numberOfMetrics)};
}

/** Value of a metric as a double (the OTF2 metric values are int64, uint64 or double). */
double metric_value_as_double(const pallas::MetricValue& v) {
    switch (v.type) {
    case pallas::PALLAS_TYPE_INT64:
        return static_cast<double>(v.value.int64);
    case pallas::PALLAS_TYPE_UINT64:
        return static_cast<double>(v.value.uint64);
    case pallas::PALLAS_TYPE_DOUBLE:
        return v.value.float64;
    case pallas::PALLAS_TYPE_FLOAT:
        return v.value.float32;
    default:
        return std::numeric_limits<double>::quiet_NaN();
    }
}

}  // namespace

std::map<pallas::MetricMemberRef, PyMetricMember> Archive_get_metric_members(pallas::Archive& archive) {
    auto map = std::map<pallas::MetricMemberRef, PyMetricMember>();
    for (auto& [key, m] : archive.definitions.metric_members) {
        map.insert(std::pair(key, make_metric_member(archive, m)));
    }
    return map;
}

std::map<pallas::MetricRef, PyMetric> Archive_get_metrics(pallas::Archive& archive) {
    auto map = std::map<pallas::MetricRef, PyMetric>();
    for (auto& [key, m] : archive.definitions.metrics) {
        map.insert(std::pair(key, make_metric(m)));
    }
    return map;
}

std::map<pallas::MetricMemberRef, PyMetricMember> Trace_get_metric_members(pallas::GlobalArchive& trace) {
    auto map = std::map<pallas::MetricMemberRef, PyMetricMember>();
    for (auto& [key, m] : trace.definitions.metric_members) {
        map.insert(std::pair(key, make_metric_member(trace, m)));
    }
    return map;
}

std::map<pallas::MetricRef, PyMetric> Trace_get_metrics(pallas::GlobalArchive& trace) {
    auto map = std::map<pallas::MetricRef, PyMetric>();
    for (auto& [key, m] : trace.definitions.metrics) {
        map.insert(std::pair(key, make_metric(m)));
    }
    return map;
}

py::list MetricValues_to_list(uint8_t nb_values, const pallas::MetricValue* values) {
    py::list list;
    for (uint8_t i = 0; i < nb_values; i++) {
        switch (values[i].type) {
        case pallas::PALLAS_TYPE_INT64:
            list.append(values[i].value.int64);
            break;
        case pallas::PALLAS_TYPE_UINT64:
            list.append(values[i].value.uint64);
            break;
        default:
            list.append(metric_value_as_double(values[i]));
            break;
        }
    }
    return list;
}

py::dict threadGetMetrics(pallas::Thread& thread) {
    struct Series {
        std::vector<std::pair<uint64_t, double>> samples;
    };
    std::map<pallas::MetricMemberRef, Series> series;

    for (size_t e = 0; e < thread.nb_events; e++) {
        auto& event = thread.events[e];
        if (event.data.record != pallas::PALLAS_EVENT_METRIC)
            continue;
        pallas::MetricRef metric_ref;
        uint8_t nb_values = 0;
        pallas::MetricValue* values = nullptr;
        pallas::pallas_read_metric(&event.data, nullptr, &metric_ref, &nb_values, &values);
        const pallas::Metric* metric = thread.archive->getMetric(metric_ref);
        for (size_t occ = 0; occ < event.nb_occurrences; occ++) {
            uint64_t ts = event.timestamps->at(occ);
            for (uint8_t i = 0; i < nb_values; i++) {
                pallas::MetricMemberRef member = (metric && i < metric->numberOfMetrics) ? metric->metric_members[i] : PALLAS_METRICMEMBERREF_INVALID;
                series[member].samples.emplace_back(ts, metric_value_as_double(values[i]));
            }
        }
        delete[] values;
    }

    py::dict result;
    for (auto& [member_ref, s] : series) {
        std::sort(s.samples.begin(), s.samples.end());
        py::ssize_t n = s.samples.size();
        py::array_t<uint64_t> timestamps(n);
        py::array_t<double> values(n);
        auto t = timestamps.mutable_unchecked<1>();
        auto v = values.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < n; i++) {
            t(i) = s.samples[i].first;
            v(i) = s.samples[i].second;
        }
        const pallas::MetricMember* m = thread.archive->getMetricMember(member_ref);
        std::string name = m ? string_or_empty(*thread.archive, m->name) : std::string();
        if (name.empty())
            name = "metric_member_" + std::to_string(member_ref);
        result[py::str(name)] = py::make_tuple(timestamps, values);
    }
    return result;
}

py::list* Trace_get_archives(pallas::GlobalArchive& trace) {
    auto* list = new py::list(trace.location_groups.size());
    int i = 0;
    for (auto& locationGroup : trace.location_groups) {
        list->operator[](i++) = trace.getArchive(locationGroup.id);
    }
    return list;
}

pallas::GlobalArchive* open_trace(const std::string& path) {
    return pallas_open_trace(path.c_str());
}

std::vector<PySequence> threadGetSequences(pallas::Thread& self) {
    auto output = std::vector<PySequence>(self.nb_sequences);
    for (size_t i = 0; i < self.nb_sequences; i++) {
        output[i].self = &self.sequences[i];
        output[i].thread = &self;
    }
    return output;
}

std::vector<PyLoop> threadGetLoops(pallas::Thread& self) {
    auto output = std::vector<PyLoop>(self.nb_loops);
    for (size_t i = 0; i < self.nb_loops; i++) {
        output[i].self = &self.loops[i];
        output[i].thread = &self;
    }
    return output;
}

std::vector<PyEvent> threadGetEvents(pallas::Thread& self) {
    auto output = std::vector<PyEvent>(self.nb_events);
    for (size_t i = 0; i < self.nb_events; i++) {
        output[i].self = &self.events[i];
        output[i].thread = &self;
    }
    return output;
}

pybind11::list sequenceGetContent(const PySequence& self) {
    auto output = pybind11::list();
    // auto output = std::vector<PySomething>(self.self->tokens.size());
    for (size_t i = 0; i < self.self->tokens.size(); i++) {
        auto t = self.self->tokens[i];
        if (t.type == pallas::TypeEvent) {
            PyEvent temp = {self.thread->getEvent(t), self.thread};
            output.append(temp);
        }
        if (t.type == pallas::TypeSequence) {
            PySequence temp = {self.thread->getSequence(t), self.thread};
            output.append(temp);
        }
        if (t.type == pallas::TypeLoop) {
            PyLoop temp = {self.thread->getLoop(t), self.thread};
            output.append(temp);
        }
    }
    return output;
}

bool doesSequenceContains(const PySequence& self, pallas::Token t) {
    for (auto token : self.self->tokens) {
        if (t == token) {
            return true;
        }
    }
    for (auto token : self.self->tokens) {
        if (token.type == pallas::TypeSequence) {
            PySequence temp = {self.thread->getSequence(token), self.thread};
            if (doesSequenceContains(temp, t)) {
                return true;
            }
        }
        if (token.type == pallas::TypeLoop) {
            auto* loop = self.thread->getLoop(token);
            PySequence temp = {self.thread->getSequence(loop->repeated_token), self.thread};
            if (doesSequenceContains(temp, t)) {
                return true;
            }
        }
    }
    return false;
}

std::vector<PyEvent> threadGetEventsMatching(pallas::Thread& t, pallas::Record record) {
    auto output = std::vector<PyEvent>();
    for (size_t i = 0; i < t.nb_events; i++) {
        if (t.events[i].data.record == record) {
            output.push_back({&t.events[i], &t});
        }
    }
    return output;
}

std::vector<PyEvent> threadGetEventsMatchingList(pallas::Thread& t, std::vector<pallas::Record> records) {
    auto output = std::vector<PyEvent>();
    for (size_t i = 0; i < t.nb_events; i++) {
        for (auto record : records) {
            if (t.events[i].data.record == record) {
                output.push_back({&t.events[i], &t});
                continue;
            }
        }
    }
    return output;
}

py::tuple makePyObjectFromToken(pallas::Token t, pallas::ThreadReader& thread_reader) {
    switch (t.type) {
    case pallas::TypeEvent: {
        return py::make_tuple(std::variant<PyEvent, PySequence, PyLoop, pallas::Token>(PyEvent{thread_reader.thread_trace->getEvent(t), thread_reader.thread_trace}),
                              thread_reader.getCurrentTokenCount(t));
    }
    case pallas::TypeSequence: {
        return py::make_tuple(std::variant<PyEvent, PySequence, PyLoop, pallas::Token>(PySequence{thread_reader.thread_trace->getSequence(t), thread_reader.thread_trace}),
                              thread_reader.getCurrentTokenCount(t));
    }
    case pallas::TypeLoop: {
        return py::make_tuple(std::variant<PyEvent, PySequence, PyLoop, pallas::Token>(PyLoop{thread_reader.thread_trace->getLoop(t), thread_reader.thread_trace}),
                              thread_reader.getCurrentTokenCount(t));
    }
    default: {
        // pallas::TypeInvalid
        return py::make_tuple(std::variant<PyEvent, PySequence, PyLoop, pallas::Token>(pallas::Token()), 0);
    }
    }
}

py::array_t<uint64_t> linked_vector_to_numpy(PyLinkedVector& self) {
    if (self.linked_vector) {
        py::ssize_t n = self.linked_vector->size;
        py::array_t<uint64_t> arr(n);
        auto r = arr.mutable_unchecked<1>();

        for (py::ssize_t i = 0; i < n; ++i) {
            r(i) = self.linked_vector->at(i);
        }
        return arr;
    } else if (self.linked_duration_vector) {
        py::ssize_t n = self.linked_duration_vector->size;
        py::array_t<uint64_t> arr(n);
        auto r = arr.mutable_unchecked<1>();

        for (py::ssize_t i = 0; i < n; ++i) {
            r(i) = self.linked_duration_vector->at(i);
        }
        return arr;
    }

    return py::array_t<uint64_t>(0);
}

std::vector<py::tuple> thread_reader_get_callstack(pallas::ThreadReader& self) {
    std::vector<py::tuple> res;
    res.reserve(self.currentState.current_frame_index);
    for (int i = 1; i <= self.currentState.current_frame_index; i++) {
        res.push_back(makePyObjectFromToken(self.currentState.callstack[i].callstack_iterable, self));
    }
    res.push_back(makePyObjectFromToken(self.pollCurToken(), self));
    return res;
}

int get_read_flags_from_bools(bool enter_sequence, bool enter_loop) {
    int flags = PALLAS_READ_FLAG_NONE;
    if (enter_sequence) {
        flags |= PALLAS_READ_FLAG_UNROLL_SEQUENCE;
    }
    if (enter_loop) {
        flags |= PALLAS_READ_FLAG_UNROLL_LOOP;
    }
    if (!flags) {
        flags = PALLAS_READ_FLAG_NO_UNROLL;
    }
    return flags;
}

py::dict get_attributes(PyEvent& event, size_t occurrence) {
  py::dict result;
  pallas::Archive* archive = event.thread->archive;
  auto* summary = event.self;
  pallas::AttributeList* attribute_list;
  if (event.self->attribute_buffer == nullptr)
    return result;
  if (summary->attribute_pos < summary->attribute_buffer_size) {
    attribute_list = (pallas::AttributeList*)&summary->attribute_buffer[summary->attribute_pos];
    while (attribute_list->index < occurrence) { /* move to the next attribute until we reach the needed index */
      summary->attribute_pos += attribute_list->struct_size;
      if (summary->attribute_pos >= summary->attribute_buffer_size)
	return result;
      attribute_list = (pallas::AttributeList*)&summary->attribute_buffer[summary->attribute_pos];
    }
    if (attribute_list->index > occurrence) {
      pallas_error("Error fetching attribute %zu. We went too far (cur position: %d) !\n", occurrence, attribute_list->index);
    }
  } else {
    return result;
  }
   
  byte* reading_addr = (byte*)attribute_list->attributes;
  byte* reading_end = (byte*)attribute_list + attribute_list->struct_size;
  for (int i = 0; i < attribute_list->nb_values; i++) {
    if (reading_addr >= reading_end)
      break;
    pallas::AttributeData* data = (pallas::AttributeData*)reading_addr;
    uint16_t size = data->struct_size;
    if (size == 0)
      break;
    const pallas::Attribute* attribute = archive->getAttribute(data->ref);
    if (attribute) {
      const pallas::String* name_str = archive->getString(attribute->name);
      if (name_str) {
	switch (attribute->type) {
	case pallas::PALLAS_TYPE_NONE:
	  break;
	case pallas::PALLAS_TYPE_UINT8: {
	  uint8_t v = data->value.uint8;
	  result[name_str->str] = v;
	  break;
	}
	case pallas::PALLAS_TYPE_UINT16: {
	  uint16_t v = data->value.uint16;
	  result[name_str->str] = v;
	  break;
	}
	case pallas::PALLAS_TYPE_UINT32: {
	  uint32_t v = data->value.uint32;
	  result[name_str->str] = v;
	  break;
	}
	case pallas::PALLAS_TYPE_UINT64: {
	  uint64_t v = data->value.uint64;
	  result[name_str->str] = v;
	  break;
	}
	case pallas::PALLAS_TYPE_INT8: {
	  int8_t v = data->value.int8;
	  result[name_str->str] = v;
	  break;
	}
	case pallas::PALLAS_TYPE_INT16: {
	  int16_t v = data->value.int16;
	  result[name_str->str] = v;
	  break;
	}
	case pallas::PALLAS_TYPE_INT32: {
	  int32_t v = data->value.int32;
	  result[name_str->str] = v;
	  break;
	}
	case pallas::PALLAS_TYPE_INT64: {
	  int64_t v = data->value.int64;
	  result[name_str->str] = v;
	  break;
	}
	case pallas::PALLAS_TYPE_FLOAT: {
	  float v = data->value.float32;
	  result[name_str->str] = v;
	  break;
	}
	case pallas::PALLAS_TYPE_DOUBLE: {
	  double v = data->value.float64;
	  result[name_str->str] = v;
	  break;
	}
	case pallas::PALLAS_TYPE_STRING: {
	  const pallas::String* val_str = archive->getString(data->value.string_ref);
	  if (val_str)
	    result[name_str->str] = val_str->str;
	} break;
	default:
	  // TODO : Add more attribute types
	  pallas_warn("Attributes of type %d are not yet handled in python\n", (int)attribute->type);
	  break;
	}
      }
    }
    reading_addr += size;
  }
  return result;
}
