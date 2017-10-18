# <<< Project Metadata >>>
project(reDPD
        VERSION 0.0.1
        LANGUAGES C CXX)
set(reDPD_AUTHORS      "Andrew M. James")
set(reDPD_DESCRIPTION  "C++ library for Tensors with Disk-Based Storage")
set(reDPD_URL          "https://github.com/CrawfordGroup/reDPD.git")
set(reDPD_LICENSE      "MIT")

# <<< Project Requirements >>>
cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# <<<  CMake Modules  >>>
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include (CheckCXXCompilerFlag)
include (CMakePackageConfigHelpers)
include(psi4OptionsTools)

# <<<  Default CMake options  >>>

option_with_default(CMAKE_BUILD_TYPE "Build type (Release or Debug)" Debug)  #Change!
option(ENABLE_TESTS         "Compile the tests"                       OFF)  #Add tests!
option(WITH_MPI             "Build the library with MPI (if avalible)"              OFF) #later
option(BUILD_FPIC           "Static library in STATIC_ONLY will be compiled with position independent code" ON)
option_with_print(ENABLE_OPENMP "Enable OpenMP parallelization" ON)
option_with_flags(ENABLE_XHOST "Enables processor-specific optimization" ON
                  "-xHost" "-march=native")
option_with_default(BUILD_FPIC "Compile static libraries with position independent code" ON)
option_with_default(CMAKE_INSTALL_LIBDIR "Directory to which libraries installed" lib)
option_with_default(ENABLE_GENERIC "Enables mostly static linking of system and math libraries for shared library" OFF)
option_with_print(BLAS_ADD_RPATH "Adds the rpath blas/Lapack libs to the rpath on the linker line" ON)

########################  Process & Validate Options  ##########################
include(GNUInstallDirs)
include(autocmake_safeguards)
include(autocmake_omp)
include(custom_cxxstandard)
include(custom_static_library)
include(custom_color_messages)

find_package(Threads REQUIRED)
link_libraries("${CMAKE_THREAD_LIBS_INIT}")

# BLAS and LAPACK
find_package(TargetLAPACK REQUIRED)

# Hacked this together, halp plz
if(BLAS_ADD_RPATH)
  SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

  # when building, don't use the install RPATH already
  # (but later on when installing)
  SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

  #set(the_install_rpath "${CMAKE_INSTALL_PREFIX}/lib")
  # add the automatically determined parts of the RPATH
  # which point to directories outside the build tree to the install RPATH
  SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

  get_property(_ill_1 TARGET tgt::lapk PROPERTY INTERFACE_LINK_LIBRARIES)
  get_property(_ill_2 TARGET tgt::blas PROPERTY INTERFACE_LINK_LIBRARIES)
  set(_all_libs "${_ill_1};${_ill_2}")
  message(STATUS "${Blue}[rpath]${Red}: Allpaths = ${_all_libs}${ColourReset}")
  foreach(_libpath  ${_all_libs})
    message(STATUS "${Blue}[rpath]:${Red} Working on: ${_libpath}${ColourReset}")
    if(${_libpath} MATCHES "\.${CMAKE_STATIC_LIBRARY_SUFFIX}$")
      message(STATUS "${Blue}[rpath]${Red} excluding path for static lib:${ColourReset} ${_libpath}")
      continue()
    endif()
    if(${_libpath} MATCHES "\.${CMAKE_SHARED_LIBRARY_SUFFIX}$")
      message(STATUS ${Blue}[rpath]${Red} "PATH: ${_libpath} is a library and willl be processed${ColourReset}")
      get_filename_component(_libdir "${_libpath}" PATH)
      list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES ${_libdir} _in_cpild)
      if(_in_cpild GREATER -1)
        message(STATUS "${Blue}[rpath]${Red} excluding system path:${ColourReset} ${_libdir}")
        continue()
      else()
        message(STATUS "${Blue}[rpah]${Green}adding path:${ColourReset} ${_libdir}")
        LIST(APPEND the_install_rpath "${_libdir}")
      endif()
    else()
      message(STATUS "${Blue}[rpath]${Red} component ${Green}${_libpath}${Red} is not a path and won't be processed${ColourReset}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES the_install_rpath)
  set(CMAKE_INSTALL_RPATH "${the_install_rpath}")
  message(STATUS "${Green}Rpath used will be:${ColourReset} ${the_install_rpath}")
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# recursively add source directories
include_directories(src)
add_subdirectory(src)
if(ENABLE_TESTS)
  add_subdirectory(samples/xmtensor_example)
else()
  message(STATUS "${Red} ==>>> REMEMBER TO SET UP TESTS YOU FOOL <<<===${ColourReset}")
endif()

# <<<  Export Config  >>>

include(CMakePackageConfigHelpers)
# GNUInstallDirs "DATADIR" wrong here; CMake search path wants "share".
set(CMAKECONFIG_INSTALL_DIR "share/cmake/${PROJECT_NAME}")
configure_package_config_file(cmake/${PROJECT_NAME}Config.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        INSTALL_DESTINATION ${CMAKECONFIG_INSTALL_DIR})
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        VERSION ${${PROJECT_NAME}_VERSION}
        COMPATIBILITY AnyNewerVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION ${CMAKECONFIG_INSTALL_DIR})
