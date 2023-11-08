#!/bin/bash

# Update and upgrade Termux packages
pkg update && pkg upgrade

# Install required packages
pkg install python clang cmake libjpeg-turbo libpng python numpy

# Install additional dependencies
pkg install ffmpeg libtiff libwebp

# Create a directory for OpenCV installation
mkdir -p ~/opencv && cd ~/opencv

# Download OpenCV and OpenCV Contrib source code
pkg install wget
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip

# Unzip the downloaded files
unzip opencv.zip
unzip opencv_contrib.zip

# Rename directories
mv opencv-4.x opencv
mv opencv_contrib-4.x opencv_contrib

# Create a build directory
mkdir -p ~/opencv/opencv/build && cd ~/opencv/opencv/build

# Configure the build with CMake
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=$PREFIX \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv/opencv_contrib/modules \
      -D BUILD_DOCS=OFF \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_opencv_python2=OFF \
      -D BUILD_opencv_python3=ON \
      -D PYTHON3_EXECUTABLE=$(which python) \
      -D PYTHON3_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
      -D PYTHON3_LIBRARY=$(python -c "import os; from distutils.sysconfig import get_config_var; print(os.path.join(os.path.dirname(get_config_var('LIBDIR')), 'libpython' + get_config_var('VERSION') + '.so'))") \
      -D PYTHON3_NUMPY_INCLUDE_DIRS=$(python -c "import numpy; print(numpy.get_include())") \
      -D PYTHON3_PACKAGES_PATH=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
      ..

# Compile OpenCV
make -j$(nproc)

# Install OpenCV
make install

# Verify OpenCV installation
python -c "import cv2; print(cv2.__version__)"
