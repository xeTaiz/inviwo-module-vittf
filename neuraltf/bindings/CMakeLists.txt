set(HEADER_FILES
    include/pyneuraltf.h
)
ivw_group("Header Files" BASE ${CMAKE_CURRENT_SOURCE_DIR}/include ${HEADER_FILES})

set(SOURCE_FILES
    src/pyneuraltf.cpp
)
ivw_group("Source Files" ${SOURCE_FILES})

ivw_add_py_wrapper(ivwneuraltf ${SOURCE_FILES} ${HEADER_FILES})
target_link_libraries(ivwneuraltf PUBLIC inviwo::module::neuraltf inviwo::module::python3)
target_include_directories(ivwneuraltf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

ivw_default_install_targets(ivwneuraltf)
