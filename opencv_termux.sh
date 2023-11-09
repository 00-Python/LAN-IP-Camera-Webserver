#!/bin/sh

# Update Termux packages
pkg update && pkg upgrade

# Install required packages
pkg install -y wget clang python ndk-sysroot cmake

# Download OpenCV for Android
mkdir -p ~/opencv-android
cd ~/opencv-android
wget -O opencv.zip https://github.com/opencv/opencv/archive/master.zip

# Unzip the OpenCV archive
unzip opencv.zip
mv opencv-master opencv

# Set up environment variables
export ANDROID_HOME=$HOME/android-sdk
export NDK_HOME=$PREFIX/libexec/android-sdk/ndk-bundle
export PATH=$PATH:$ANDROID_HOME/tools:$ANDROID_HOME/platform-tools:$NDK_HOME

# Download Android SDK (command line tools)
mkdir -p $ANDROID_HOME
cd $ANDROID_HOME
wget -O sdk-tools.zip https://dl.google.com/android/repository/commandlinetools-linux-6858069_latest.zip
unzip sdk-tools.zip -d cmdline-tools
mv cmdline-tools tools

# Accept licenses
yes | tools/bin/sdkmanager --licenses

# Install SDK platforms and build tools
tools/bin/sdkmanager "platform-tools" "platforms;android-29" "build-tools;29.0.3"

# Download and set up the Android NDK
cd $NDK_HOME
wget -O ndk.zip https://dl.google.com/android/repository/android-ndk-r21e-linux-x86_64.zip
unzip ndk.zip

# Build OpenCV for Android
cd ~/opencv-android/opencv
mkdir build
cd build
cmake -D CMAKE_TOOLCHAIN_FILE=$NDK_HOME/build/cmake/android.toolchain.cmake \
      -D ANDROID_NDK=$NDK_HOME \
      -D ANDROID_NATIVE_API_LEVEL=21 \
      -D ANDROID_ABI=arm64-v8a \
      -D ANDROID_STL=c++_shared \
      ..
make -j4

# The OpenCV Android library is now built and ready to be included in Android projects
echo "OpenCV for Android is ready!"
