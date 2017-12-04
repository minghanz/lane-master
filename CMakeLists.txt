cmake_minimum_required(VERSION 2.8)
project( lane-detection )
find_package( OpenCV 3.3.0 REQUIRED )
find_package(Armadillo REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} 
${ARMADILLO_INCLUDE_DIRS} )

add_subdirectory(alglib)

add_executable( lane-detection main.cpp LaneImage.cpp LaneImage_fit.cpp LaneImage_p0.cpp LaneImage_pp.cpp Line.cpp VanPt.cpp LearnModel.cpp LaneMark.cpp VehMask.cpp KeyPts.cpp)

target_link_libraries( lane-detection ${OpenCV_LIBS}
${ARMADILLO_LIBRARIES} alglib
)


add_definitions(-std=c++11)
