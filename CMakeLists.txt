cmake_minimum_required(VERSION 3.10.2)
set (CMAKE_CXX_STANDARD 17)

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

project(kandi3dloc)

set(SOURCES main.cpp 
    src/camera_module.h src/camera_module.cpp
    src/localization_graph.h src/localization_graph.cpp
    src/features.h
    src/frame.h
    src/constants.h
    src/debug_functions.h
    src/utilities.h
)


find_package(Open3D HINTS ${CMAKE_INSTALL_PREFIX}/lib/CMake)
list(APPEND Open3D_LIBRARIES dl)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${Open3D_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Open3D_CXX_FLAGS} -fopenmp") 
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${Open3D_EXE_LINKER_FLAGS}")

add_definitions(-DUNIX)
add_compile_options(-Wno-deprecated-declarations)
add_compile_options(-Wno-unused-result)

# add_definitions(-g -fopenmp)
add_definitions(-O3 -fopenmp)

message(STATUS "Found Open3D ${Open3D_VERSION} ${Open3D_LIBRARY_DIRS}")

# find OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

link_directories(${Open3D_LIBRARY_DIRS})

include_directories( ${EIGEN_INCLUDE_DIR})

add_executable(kandi3dloc ${SOURCES})

target_link_libraries(kandi3dloc 
                        ${Open3D_LIBRARIES} 
                        ${OpenCV_LIBS}
                        Eigen3::Eigen
)
