ivw_module(NeuralTF)

set(HEADER_FILES
    include/inviwo/neuraltf/bindings/pyneuraltf.h
    include/inviwo/neuraltf/neuraltfmodule.h
    include/inviwo/neuraltf/neuraltfmoduledefine.h
    include/inviwo/neuraltf/processors/brushvolumeslicegl.h
    include/inviwo/neuraltf/processors/dinovolumerenderer.h
    include/inviwo/neuraltf/properties/ntfproperty.h
)
ivw_group("Header Files" ${HEADER_FILES})

set(SOURCE_FILES
    src/bindings/pyneuraltf.cpp
    src/neuraltfmodule.cpp
    src/processors/brushvolumeslicegl.cpp
    src/processors/dinovolumerenderer.cpp
    src/properties/ntfproperty.cpp
)
ivw_group("Source Files" ${SOURCE_FILES})

set(SHADER_FILES
    glsl/image_identity.vert
    glsl/neuraltfraycaster.frag
    glsl/annotation.frag
)
ivw_group("Shader Files" ${SHADER_FILES})
set(PYTHON_FILES
    python/processors/DinoSimilarities.py
    python/processors/NumpyVolumeSource.py
    python/processors/SegmentationVolumeSplitter.py
    python/processors/volume_numpy_export.py
    python/bilateral_solver3d.py
    python/infer.py
)
ivw_group("Python Files" ${PYTHON_FILES})

set(TEST_FILES
    tests/unittests/neuraltf-unittest-main.cpp
)
ivw_add_unittest(${TEST_FILES})

ivw_create_module(${SOURCE_FILES} ${HEADER_FILES} ${SHADER_FILES} ${PYTHON_FILES})

# Add shader directory to install package
ivw_add_to_module_pack(${CMAKE_CURRENT_SOURCE_DIR}/glsl)
ivw_add_to_module_pack(${CMAKE_CURRENT_SOURCE_DIR}/python)
ivw_add_to_module_pack(${CMAKE_CURRENT_SOURCE_DIR}/python/processors)

find_package(pybind11 CONFIG REQUIRED)
target_link_libraries(inviwo-module-neuraltf PUBLIC pybind11::embed Python3::Python)
