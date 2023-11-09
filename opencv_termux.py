import os
import subprocess

# Function to run a shell command and print its output
def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error: {stderr.decode('utf-8')}")
    else:
        print(stdout.decode('utf-8'))

# Update packages and install required dependencies
print("Updating packages and installing required dependencies...")
run_command("pkg update && pkg upgrade")
run_command("pkg install clang python cmake libjpeg-turbo ndk-sysroot ndk-stl ndk-multilib make")

# Set environment variables
home_dir = os.getenv("HOME")
android_ndk_root = f"{home_dir}/android-ndk-r4-crystax"
android_sdk_root = f"{home_dir}/Android/android-sdk-linux"
ant_home = f"{home_dir}/Android/apache-ant-1.8.4"
java_home = f"{home_dir}/jdk1.7.0_03"
opencv_root = f"{home_dir}/Android/opencv"

env_vars = {
    "NDK": android_ndk_root,
    "SDK": android_sdk_root,
    "ANT_HOME": ant_home,
    "JAVA_HOME": java_home,
    "OPCV": opencv_root,
    "PATH": f"{android_ndk_root}:{android_sdk_root}/tools:{android_sdk_root}/platform-tools:{ant_home}/bin:{java_home}/bin:${{PATH}}"
}

# Export environment variables
print("Exporting environment variables...")
for key, value in env_vars.items():
    os.environ[key] = value
    run_command(f"echo 'export {key}={value}' >> ~/.bashrc")

# Source the bashrc file
run_command("source ~/.bashrc")

# Download and install OpenCV
print("Downloading and installing OpenCV...")
run_command(f"wget http://ee368.stanford.edu/Android/OpenCV/opencv.tar.gz -O {home_dir}/opencv.tar.gz")
run_command(f"tar -xvf {home_dir}/opencv.tar.gz -C {home_dir}/Android")

# Build OpenCV
print("Building OpenCV...")
run_command(f"cd {opencv_root}/android && mkdir build && cd build && cmake .. && make")

# Download and install CVCamera_MSER
print("Downloading and installing CVCamera_MSER...")
run_command(f"wget http://ee368.stanford.edu/Android/OpenCV/CVCamera_MSER.zip -O {home_dir}/CVCamera_MSER.zip")
run_command(f"unzip {home_dir}/CVCamera_MSER.zip -d {opencv_root}/android/apps")

# Build CVCamera_MSER
print("Building CVCamera_MSER...")
run_command(f"cd {opencv_root}/android/apps/CVCamera_MSER && sh project_create.sh && make clean && make V=0 && ant clean && ant debug")

print("Installation script completed.")
