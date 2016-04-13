cmake_minimum_required(VERSION 2.8.7)

project( SaliencyELD )

# Tested on g++ 4.9.2
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fopenmp -std=c++11")

# VLfeat library setting (NOTE: Change to your vlfeat path)
include_directories(./library/vlfeat-0.9.20)
link_directories(./library/vlfeat-0.9.20/bin/glnxa64)

# Caffe setting
# symlink your caffe to this directory. (NOTE: Build Caffe with Cmake first)
set(Caffe_DIR ./caffe/build)
find_package(Caffe REQUIRED NO_DEFAULT_PATH)
include_directories(./caffe/build/include)
include_directories(./caffe/include)
include_directories(${Caffe_INCLUDE_DIRS})

# Boost
find_package(Boost 1.46 REQUIRED COMPONENTS system thread filesystem chrono)

# Tested on OpenCV 3.0
# You should remove imgcodecs and change some include path if you use opencv 2.x
find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs contrib)

#    Find all source files
set(executable SaliencyELD)
file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_executable(${executable}.bin ${SOURCES})
set_target_properties(${executable}.bin PROPERTIES OUTPUT_NAME ${executable})
target_link_libraries(${executable}.bin ${Caffe_LIBRARIES} -lvl ${Boost_LIBRARIES} ${OpenCV_LIBS})
