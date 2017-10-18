add_subdirectory(libxm)
set(TENSOR_SOURCES xmtensor.cc)
set(TENSOR_HEADERS xmtensor.h local_types.h)
add_library(reDPD STATIC ${TENSOR_SOURCES} ${TENSOR_HEADERS})
set_target_properties(reDPD PROPERTIES 
                      POSITION_INDEPENDENT_CODE ${BUILD_FPIC}
                      OUTPUT_NAME "reDPD"
                      EXPORT_NAME "reDPD")
target_link_libraries(reDPD libxm)
install(TARGETS reDPD
        EXPORT "${PROJECT_NAME}Targets"
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
