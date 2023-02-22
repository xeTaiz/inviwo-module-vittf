ivw_module(NeuralTF)

set(HEADER_FILES
    include/inviwo/neuraltf/neuraltfmodule.h
    include/inviwo/neuraltf/neuraltfmoduledefine.h
    include/inviwo/neuraltf/processors/dinovolumerenderer.h
    include/inviwo/neuraltf/properties/ntfproperty.h
)
ivw_group("Header Files" ${HEADER_FILES})

set(SOURCE_FILES
    src/neuraltfmodule.cpp
    src/processors/dinovolumerenderer.cpp
    src/properties/ntfproperty.cpp
)
ivw_group("Source Files" ${SOURCE_FILES})

set(SHADER_FILES
    glsl/image_identity.vert
    glsl/neuraltfraycaster.frag
)
ivw_group("Shader Files" ${SHADER_FILES})

set(TEST_FILES
    tests/unittests/neuraltf-unittest-main.cpp
)
ivw_add_unittest(${TEST_FILES})

ivw_create_module(${SOURCE_FILES} ${HEADER_FILES} ${SHADER_FILES})

# Add shader directory to install package
ivw_add_to_module_pack(${CMAKE_CURRENT_SOURCE_DIR}/glsl)
