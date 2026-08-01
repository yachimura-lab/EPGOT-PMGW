cmake_minimum_required(VERSION 3.12)
project(EntropicPartialOT)

set(CMAKE_CXX_STANDARD 17)

# 寻找依赖
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

# 定义模块：源文件就在当前目录
pybind11_add_module(cpp_engine engine.cpp)

# 链接 Eigen
target_link_libraries(cpp_engine PRIVATE Eigen3::Eigen)

# 关键修正：确保输出到项目根目录 (project/)
# PROJECT_SOURCE_DIR 通常指向最顶层的 CMakeLists，但在你目前的结构中
# 使用 CMAKE_CURRENT_SOURCE_DIR/.. 是最保险的
set_target_properties(cpp_engine PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
