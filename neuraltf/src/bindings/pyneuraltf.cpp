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
        .def("clearAnnotations", &NTFProperty::clearAnnotations)
        .def("setAnnotations", &NTFProperty::setAnnotations)
        .def("getAnnotatedVoxels", &NTFProperty::getAnnotatedVoxels)
        .def_property_readonly("tf", py::cpp_function([](NTFProperty& p) -> TransferFunctionProperty& { return p.tf_; },
                             py::return_value_policy::reference_internal))
        .def_property("modality", &NTFProperty::getModality, &NTFProperty::setModality)
        .def_property("modalityWeight", &NTFProperty::getModalityWeight, &NTFProperty::setModalityWeight)
        .def_property("isoValue", &NTFProperty::getIsoValue, &NTFProperty::setIsoValue)
        .def_property("color", &NTFProperty::getColor, &NTFProperty::setColor)
        .def_property("similarityReduction", &NTFProperty::getSimilarityReduction, &NTFProperty::setSimilarityReduction)
        .def_property("requiresUpdate", &NTFProperty::getRequiresUpdate, &NTFProperty::setRequiresUpdate)
        .def_property("blsEnabled", &NTFProperty::blsEnabled, &NTFProperty::enableBLS)
        .def_property("blsSigmas", &NTFProperty::getBLSSigma, &NTFProperty::setBLSSigma);
}

}  // namespace inviwo
