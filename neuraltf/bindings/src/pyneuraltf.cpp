#include <pyneuraltf.h>

#include <inviwo/neuraltf/properties/ntfproperty.h>
#include <inviwo/core/properties/compositeproperty.h>

namespace py = pybind11;

namespace inviwo {

void exposeNTFBindings(py::module& m) {
        py::class_<NTFProperty, CompositeProperty>(m, "NTFProperty")
        .def(py::init([](std::string_view identifier, std::string_view displayName, VolumeInport* inport, 
                InvalidationLevel invalidationLevel, PropertySemantics semantics) {
                 return new NTFProperty(identifier, displayName, inport, invalidationLevel, semantics);
             }),
             py::arg("identifier"), py::arg("displayName"),
             py::arg("inport") = nullptr,
             py::arg("invalidationLevel") = InvalidationLevel::InvalidResources,
             py::arg("semantics") = PropertySemantics::Default)
        .def("addAnnotation", &NTFProperty::addAnnotation)
        .def("removeAnnotation", &NTFProperty::removeAnnotation)
        .def("getAnnotatedVoxels", &NTFProperty::getAnnotatedVoxels)
        .def_readonly("tf", &NTFProperty::tf_)
        .def_readonly("simtf", &NTFProperty::simTf_)
        .def_readonly("modality", &NTFProperty::modality_)
        .def_readonly("modalityWeight", &NTFProperty::modalityWeight_)
        .def_readonly("similarityExponent", &NTFProperty::similarityExponent_)
        .def_readonly("similarityThreshold", &NTFProperty::similarityThreshold_)
        .def_readonly("volumeInport", &NTFProperty::volumeInport_);
}

}  // namespace inviwo
