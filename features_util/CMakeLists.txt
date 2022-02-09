cmake_minimum_required(VERSION 3.0.2)
project(features_util)

add_compile_options(-std=c++17)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(Eigen3 REQUIRED)


find_package(OpenMP)
if (OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


###########
## Build ##
###########

add_subdirectory(thirdparty/pybind)

# Features Util
pybind11_add_module(features_util_cpp features_util_cpp.cpp)
target_compile_options(features_util_cpp PRIVATE -std=c++17)
target_include_directories(features_util_cpp PUBLIC include ${EIGEN3_INCLUDE_DIR} thirdparty/nanoflann/include)
target_link_libraries(features_util_cpp PRIVATE )