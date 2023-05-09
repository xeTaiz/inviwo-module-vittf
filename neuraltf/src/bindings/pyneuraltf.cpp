#include <inviwo/neuraltf/bindings/pyneuraltf.h>

#include <inviwo/neuraltf/properties/ntfproperty.h>
#include <inviwo/core/properties/compositeproperty.h>

#include <pybind11/stl.h>
#include <pybind11/functional.h>

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
        .def_property_readonly("tf", py::cpp_function([](NTFProperty& p) -> TransferFunctionProperty& { return p.tf_; },
                             py::return_value_policy::reference_internal))
        .def_property_readonly("simtf", py::cpp_function([](NTFProperty& p) -> TransferFunctionProperty& { return p.simTf_; },
                             py::return_value_policy::reference_internal))
        .def_property("modality", &NTFProperty::getModality, &NTFProperty::setModality)
        .def_property("modalityWeight", &NTFProperty::getModalityWeight, &NTFProperty::setModalityWeight)
        .def_property("similarityExponent", &NTFProperty::getSimilarityExponent, &NTFProperty::setSimilarityExponent)
        .def_property("similarityThreshold", &NTFProperty::getSimilarityThreshold, &NTFProperty::setSimilarityThreshold)
        .def_property("similarityReduction", &NTFProperty::getSimilarityReduction, &NTFProperty::setSimilarityReduction)
        .def_property("requiresUpdate", &NTFProperty::getRequiresUpdate, &NTFProperty::setRequiresUpdate);
}

}  // namespace inviwo
