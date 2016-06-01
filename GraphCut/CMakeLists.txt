cmake_minimum_required (VERSION 2.8)
project (GraphCut)

set (CMAKE_CXX_STANDARD 11)

add_compile_options(-Werror -Wextra -O3)

FIND_LIBRARY(OPENCV opencv_core)
FIND_LIBRARY(OPENCV_GUI opencv_highgui)

SET(EXTRA_LIBS ${OPENCV} ${OPENCV_GUI})

FILE(GLOB ALL *.cpp)

set(SRCS ${UTIL} ${ALL})
add_executable(GraphCut ${SRCS})

TARGET_LINK_LIBRARIES(GraphCut ${EXTRA_LIBS})