set(XM_SOURCES alloc.c blockspace.c contract.c dim.c scalar.c tensor.c util.c xm.c)
set(XM_HEADERS alloc.h blockspace.h bitmap.h dim.h scalar.h tensor.h util.h xm.h)
add_library(libxm STATIC ${XM_SOURCES} ${XM_HEADERS})
set_target_properties(libxm PROPERTIES
                      POSITION_INDEPENDENT_CODE ${BUILD_FPIC}
                      OUTPUT_NAME "libxm"
                      EXPORT_NAME "libxm"
                      )
target_link_libraries(libxm tgt::lapack)
