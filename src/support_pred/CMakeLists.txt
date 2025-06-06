cmake_minimum_required(VERSION 3.16)
project(halfplane_module)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Required for CGAL
find_package(CGAL REQUIRED COMPONENTS Core)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

include_directories(include)

# To Debug
# add_compile_definitions(PYBIND11_DETAILED_ERROR_MESSAGES)

# Create the module
pybind11_add_module(halfplane_module
    src/HalfplaneIntersectionProcessor.cpp
    src/Logger.cpp
)
# strip any “lib” prefix and force .so suffix
set_target_properties(halfplane_module PROPERTIES
  PREFIX ""
  SUFFIX ".so"
)

# Link CGAL libraries
target_link_libraries(
    halfplane_module PRIVATE
    CGAL::CGAL
    CGAL::CGAL_Core
)


