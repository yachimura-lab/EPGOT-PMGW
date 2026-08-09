cmake_minimum_required(VERSION 3.22.2)
project(EntropicPartialOT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PYBIND11_FINDPYTHON ON)

find_package(
    Python
    COMPONENTS Interpreter Development.Module
    REQUIRED
)

find_package(pybind11 3.0.4 EXACT CONFIG REQUIRED)
find_package(Eigen3 3.3.9 EXACT REQUIRED)

pybind11_add_module(cpp_engine MODULE engine.cpp)

target_link_libraries(
    cpp_engine
    PRIVATE
        Eigen3::Eigen
)

# Emit the module into src/, which is on the notebooks' sys.path
get_filename_component(
    CPP_ENGINE_OUTPUT_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    ABSOLUTE
)

set_target_properties(
    cpp_engine
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CPP_ENGINE_OUTPUT_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${CPP_ENGINE_OUTPUT_DIR}"
)

foreach(CONFIG DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(
        cpp_engine
        PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY_${CONFIG} "${CPP_ENGINE_OUTPUT_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_${CONFIG} "${CPP_ENGINE_OUTPUT_DIR}"
    )
endforeach()

message(STATUS "Python executable: ${Python_EXECUTABLE}")
message(STATUS "Python version: ${Python_VERSION}")
message(STATUS "cpp_engine output directory: ${CPP_ENGINE_OUTPUT_DIR}")
