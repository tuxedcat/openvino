// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include <pybind11/stl.h>

#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "Python.h"
#include "openvino/frontend/decoder.hpp"

using Version = ov::pass::Serialize::Version;

namespace Common {
namespace utils {

py::object from_ov_any(const ov::Any& any) {
    // Check for py::object
    if (any.is<py::object>()) {
        return any.as<py::object>();
    }
    // Check for std::string
    else if (any.is<std::string>()) {
        return py::cast(any.as<std::string>().c_str());
    }
    // Check for int
    else if (any.is<int>()) {
        return py::cast(any.as<int>());
    } else if (any.is<int64_t>()) {
        return py::cast(any.as<int64_t>());
    }
    // Check for unsigned int
    else if (any.is<unsigned int>()) {
        return py::cast(any.as<unsigned int>());
    } else if (any.is<uint64_t>()) {
        return py::cast(any.as<uint64_t>());
    }
    // Check for float
    else if (any.is<float>()) {
        return py::cast(any.as<float>());
    } else if (any.is<double>()) {
        return py::cast(any.as<double>());
    }
    // Check for bool
    else if (any.is<bool>()) {
        return py::cast(any.as<bool>());
    }
    // Check for std::vector<std::string>
    else if (any.is<std::vector<std::string>>()) {
        return py::cast(any.as<std::vector<std::string>>());
    }
    // Check for std::vector<int>
    else if (any.is<std::vector<int>>()) {
        return py::cast(any.as<std::vector<int>>());
    }
    // Check for std::vector<int64_t>
    else if (any.is<std::vector<int64_t>>()) {
        return py::cast(any.as<std::vector<int64_t>>());
    }
    // Check for std::vector<unsigned int>
    else if (any.is<std::vector<unsigned int>>()) {
        return py::cast(any.as<std::vector<unsigned int>>());
    }
    // Check for std::vector<float>
    else if (any.is<std::vector<float>>()) {
        return py::cast(any.as<std::vector<float>>());
    }
    // Check for std::vector<double>
    else if (any.is<std::vector<double>>()) {
        return py::cast(any.as<std::vector<double>>());
    }
    // Check for std::tuple<unsigned int, unsigned int>
    else if (any.is<std::tuple<unsigned int, unsigned int>>()) {
        return py::cast(any.as<std::tuple<unsigned int, unsigned int>>());
    }
    // Check for std::tuple<unsigned int, unsigned int, unsigned int>
    else if (any.is<std::tuple<unsigned int, unsigned int, unsigned int>>()) {
        return py::cast(any.as<std::tuple<unsigned int, unsigned int, unsigned int>>());
    }
    // Check for std::map<std::string, std::string>
    else if (any.is<std::map<std::string, std::string>>()) {
        return py::cast(any.as<std::map<std::string, std::string>>());
    }
    // Check for std::map<std::string, int>
    else if (any.is<std::map<std::string, int>>()) {
        return py::cast(any.as<std::map<std::string, int>>());
    }
    // Check for std::map<std::string, uint64_t>
    else if (any.is<std::map<std::string, uint64_t>>()) {
        return py::cast(any.as<std::map<std::string, uint64_t>>());
    }
    // Check for std::map<element::Type, float>
    else if (any.is<std::map<ov::element::Type, float>>()) {
        return py::cast(any.as<std::map<ov::element::Type, float>>());
    }
    // Check for std::vector<ov::PropertyName>
    else if (any.is<std::vector<ov::PropertyName>>()) {
        auto val = any.as<std::vector<ov::PropertyName>>();
        PyObject* dict = PyDict_New();
        for (const auto& it : val) {
            std::string property_name = it;
            std::string mutability = it.is_mutable() ? "RW" : "RO";
            PyDict_SetItemString(dict, property_name.c_str(), PyUnicode_FromString(mutability.c_str()));
        }
        return py::cast<py::object>(dict);
    } else if (any.is<ov::element::Type>()) {
        return py::cast(any.as<ov::element::Type>());
    } else if (any.is<ov::hint::Priority>()) {
        return py::cast(any.as<ov::hint::Priority>());
    } else if (any.is<ov::hint::PerformanceMode>()) {
        return py::cast(any.as<ov::hint::PerformanceMode>());
    } else if (any.is<ov::hint::ExecutionMode>()) {
        return py::cast(any.as<ov::hint::ExecutionMode>());
    } else if (any.is<ov::log::Level>()) {
        return py::cast(any.as<ov::log::Level>());
    } else if (any.is<ov::device::Type>()) {
        return py::cast(any.as<ov::device::Type>());
    } else if (any.is<ov::streams::Num>()) {
        return py::cast(any.as<ov::streams::Num>());
    } else if (any.is<ov::Affinity>()) {
        return py::cast(any.as<ov::Affinity>());
    } else if (any.is<ov::device::UUID>()) {
        std::stringstream uuid_stream;
        uuid_stream << any.as<ov::device::UUID>();
        return py::cast(uuid_stream.str());
        // Custom FrontEnd Types
    } else if (any.is<ov::frontend::type::List>()) {
        return py::cast(any.as<ov::frontend::type::List>());
    } else if (any.is<ov::frontend::type::Tensor>()) {
        return py::cast(any.as<ov::frontend::type::Tensor>());
    } else if (any.is<ov::frontend::type::Str>()) {
        return py::cast(any.as<ov::frontend::type::Str>());
    } else if (any.is<ov::frontend::type::PyNone>()) {
        return py::cast(any.as<ov::frontend::type::PyNone>());
    } else {
        PyErr_SetString(PyExc_TypeError, "Failed to convert parameter to Python representation!");
        return py::cast<py::object>((PyObject*)NULL);
    }
}

std::map<std::string, ov::Any> properties_to_any_map(const std::map<std::string, py::object>& properties) {
    std::map<std::string, ov::Any> properties_to_cpp;
    for (const auto& property : properties) {
        properties_to_cpp[property.first] = Common::utils::py_object_to_any(property.second);
    }
    return properties_to_cpp;
}

std::string convert_path_to_string(const py::object& path) {
    // import pathlib.Path
    py::object Path = py::module_::import("pathlib").attr("Path");
    // check if model path is either a string or pathlib.Path
    if (py::isinstance(path, Path) || py::isinstance<py::str>(path)) {
        return path.str();
    }
    // Convert bytes to string
    if (py::isinstance<py::bytes>(path)) {
        return path.cast<std::string>();
    }
    std::stringstream str;
    str << "Path: '" << path << "'"
        << " does not exist. Please provide valid model's path either as a string, bytes or pathlib.Path. "
           "Examples:\n(1) '/home/user/models/model.onnx'\n(2) Path('/home/user/models/model/model.onnx')";
    throw ov::Exception(str.str());
}

Version convert_to_version(const std::string& version) {
    if (version == "UNSPECIFIED")
        return Version::UNSPECIFIED;
    if (version == "IR_V10")
        return Version::IR_V10;
    if (version == "IR_V11")
        return Version::IR_V11;
    throw ov::Exception("Invoked with wrong version argument: '" + version +
                        "'! The supported versions are: 'UNSPECIFIED'(default), 'IR_V10', 'IR_V11'.");
}

void deprecation_warning(const std::string& function_name, const std::string& version, const std::string& message) {
    std::stringstream ss;
    ss << function_name << " is deprecated";
    if (!version.empty()) {
        ss << " and will be removed in version " << version;
    }
    if (!message.empty()) {
        ss << ". " << message;
    }
    PyErr_WarnEx(PyExc_DeprecationWarning, ss.str().data(), 2);
}

ov::Any py_object_to_any(const py::object& py_obj) {
    // Python types
    if (py::isinstance<py::str>(py_obj)) {
        return py_obj.cast<std::string>();
    } else if (py::isinstance<py::bool_>(py_obj)) {
        return py_obj.cast<bool>();
    } else if (py::isinstance<py::float_>(py_obj)) {
        return py_obj.cast<double>();
    } else if (py::isinstance<py::int_>(py_obj)) {
        return py_obj.cast<int64_t>();
    } else if (py::isinstance<py::list>(py_obj)) {
        auto _list = py_obj.cast<py::list>();
        enum class PY_TYPE : int { UNKNOWN = 0, STR, INT, FLOAT, BOOL };
        PY_TYPE detected_type = PY_TYPE::UNKNOWN;
        for (const auto& it : _list) {
            auto check_type = [&](PY_TYPE type) {
                if (detected_type == PY_TYPE::UNKNOWN || detected_type == type) {
                    detected_type = type;
                    return;
                }
                OPENVINO_ASSERT("Incorrect attribute. Mixed types in the list are not allowed.");
            };
            if (py::isinstance<py::str>(it)) {
                check_type(PY_TYPE::STR);
            } else if (py::isinstance<py::int_>(it)) {
                check_type(PY_TYPE::INT);
            } else if (py::isinstance<py::float_>(it)) {
                check_type(PY_TYPE::FLOAT);
            } else if (py::isinstance<py::bool_>(it)) {
                check_type(PY_TYPE::BOOL);
            }
        }

        // In case of empty vector works like with vector of strings
        if (_list.empty())
            return _list.cast<std::vector<std::string>>();

        switch (detected_type) {
        case PY_TYPE::STR:
            return _list.cast<std::vector<std::string>>();
        case PY_TYPE::FLOAT:
            return _list.cast<std::vector<double>>();
        case PY_TYPE::INT:
            return _list.cast<std::vector<int64_t>>();
        case PY_TYPE::BOOL:
            return _list.cast<std::vector<bool>>();
        default:
            OPENVINO_ASSERT(false, "Unsupported attribute type.");
        }
        // OV types
    } else if (py::isinstance<ov::Any>(py_obj)) {
        return py::cast<ov::Any>(py_obj);
    } else if (py::isinstance<ov::element::Type>(py_obj)) {
        return py::cast<ov::element::Type>(py_obj);
    } else if (py::isinstance<ov::hint::Priority>(py_obj)) {
        return py::cast<ov::hint::Priority>(py_obj);
    } else if (py::isinstance<ov::hint::PerformanceMode>(py_obj)) {
        return py::cast<ov::hint::PerformanceMode>(py_obj);
    } else if (py::isinstance<ov::log::Level>(py_obj)) {
        return py::cast<ov::log::Level>(py_obj);
    } else if (py::isinstance<ov::device::Type>(py_obj)) {
        return py::cast<ov::device::Type>(py_obj);
    } else if (py::isinstance<ov::streams::Num>(py_obj)) {
        return py::cast<ov::streams::Num>(py_obj);
    } else if (py::isinstance<ov::Affinity>(py_obj)) {
        return py::cast<ov::Affinity>(py_obj);
        // FrontEnd Decoder
    } else if (py::isinstance<ov::frontend::IDecoder>(py_obj)) {
        return py::cast<std::shared_ptr<ov::frontend::IDecoder>>(py_obj);
        // Custom FrontEnd Types
    } else if (py::isinstance<ov::frontend::type::Tensor>(py_obj)) {
        return py::cast<ov::frontend::type::Tensor>(py_obj);
    } else if (py::isinstance<ov::frontend::type::List>(py_obj)) {
        return py::cast<ov::frontend::type::List>(py_obj);
    } else if (py::isinstance<ov::frontend::type::Str>(py_obj)) {
        return py::cast<ov::frontend::type::Str>(py_obj);
    } else if (py::isinstance<ov::frontend::type::PyNone>(py_obj)) {
        return py::cast<ov::frontend::type::PyNone>(py_obj);
        // If there is no match fallback to py::object
    } else if (py::isinstance<py::object>(py_obj)) {
        return py_obj;
    }
    OPENVINO_ASSERT(false, "Unsupported attribute type.");
}
};  // namespace utils
};  // namespace Common
