//
// Created by khatharsis on 10/02/25.
//
#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <pallas/pallas.h>
#include <pallas/pallas_archive.h>
#include <pallas/pallas_storage.h>

namespace PYBIND11_NAMESPACE {
namespace detail {
template <>
struct type_caster<pallas::String> {
  PYBIND11_TYPE_CASTER(pallas::String, const_name("pallas::String"));
  static handle cast(pallas::String src, return_value_policy /* policy */, handle /* parent */) { return PyUnicode_FromStringAndSize(src.str, src.length - 1); }
};
}  // namespace detail
}  // namespace PYBIND11_NAMESPACE