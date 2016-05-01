#!/bin/bash
# Argument: main .cpp filename zonder .cpp
# Bij eerste run nieuwe executable executie rechten geven
cd $PWD
if [ $# != 1 ]; then
	echo "Use: openCvCmake.bash [projectName]">&2
	exit 1
fi
sudo rm -rf CMake*
projectName=$1
touch $projectName
if [ ! -f CMakeLists.txt ]; then
	cat > CMakeLists.txt <<EOF
cmake_minimum_required(VERSION 2.8)
project( $projectName )
find_package( OpenCV REQUIRED )
add_executable( $projectName $projectName.cpp )
target_link_libraries( $projectName \${OpenCV_LIBS} )
EOF
fi;
if [ ! -d CMake ]; then
	mkdir CMake 1>&2 2>/dev/null
fi;
cd CMake
cmake ..
make
cp $projectName ../
cd ..
chmod u+x $projectName 