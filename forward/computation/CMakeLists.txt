cmake_minimum_required(VERSION 3.12)
project(EntropicPartialOT)

set(CMAKE_CXX_STANDARD 17)

# Find dependencies
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

# Define the module: the source file sits in this directory
pybind11_add_module(cpp_engine engine.cpp)

# Link Eigen
target_link_libraries(cpp_engine PRIVATE Eigen3::Eigen)

# Important: make sure the module is emitted into the project root (project/).
# PROJECT_SOURCE_DIR usually points at the top-level CMakeLists, but with the
# current layout using CMAKE_CURRENT_SOURCE_DIR/.. is the safest choice.
set_target_properties(cpp_engine PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
