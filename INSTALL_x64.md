# Installation on Ubuntu18.04(x64)
x64 な環境で動画ファイルの入出力のために必要なライブラリたちをインストールする。

必要なライブラリは、
* FFMpeg (ver.4.2 ~)
* GStreamer (ver.1.14 ~)
* DeepStream SDK (ver.5.1 ~)
* OpenCV (ver.4.1.1 ~)

## FFmpeg
[この Web ページ](https://sourcedigit.com/24586-ffmpeg-4-2-ada-install-ffmpeg-4-2-in-ubuntu-18-04/)や
[ここ](https://ubuntuhandbook.org/index.php/2021/05/install-ffmpeg-4-4-ppa-ubuntu-20-04-21-04/)の通りにする。
```bash
sudo apt-get update
sudo add-apt-repository ppa:jonathonf/ffmpeg-4
sudo apt-get update
sudo apt-get install ffmpeg
sudo apt install \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavresample-dev \
    libavutil-dev \
    libpostproc-dev \
    libswresample-dev \
    libswscale-dev
```

## GStreamer(JetPack ではインストール済み)
```bash
sudo apt install libgstreamer1.0-0 gstreamer1.0-tools
sudo apt install \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-doc \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudios \
    libgstreamer-plugins-bad1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev
```

## DeepStream SDK
Jetson 環境では不要だが、
x64 環境では、NVDEC/NVENC 用の GStreamer エレメントが入っていないのでインストールする。
おおよそ[dGPU Setup for Ubuntu](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html#dgpu-setup-for-ubuntu)の手順に従ってインストールする。
詳しくは、[ここ](./memo_install_Ubuntu1804_for_GamingNotePC.md#deepstream%E3%81%AE%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%BC%E3%83%AB%E6%89%8B%E9%A0%86)を参照。

### 任意でインストール
DeepStream SDK とは別の NVENC/NVDEC エレメントを
[ここ](http://lifestyletransfer.com/how-to-install-nvidia-gstreamer-plugins-nvenc-nvdec-on-ubuntu/)の通りにインストールする。
GStreamer のバージョンは、1.14.5 で少し古いので、
もしかすると、1.18 以上で自前でビルド&インストールする方が幸せかもしれない。

## OpenCV
NV12 に対応した VideoCaptureクラスのためにパッチを当ててインストールする。

パッチを当てると、
* NV12 対応
* 8~16bit までのビット長に対応
* float, half-float 対応

パッチファイルは、**OpenCV_ffmpeg_patch** ディレクトリにいくつかある。  
**opencv-*****\<version\>*****/modules/videoio/src** にある下記の 2つのファイルを置き換える。
* cap_ffmpeg.cpp
* cap_ffmpeg_impl.hpp

OpenCV のバージョンに対応したパッチファイルを使う。  

```bash
export OPENCV_VERSION='4.1.1'   # '4.1.1', '4.4.0', ...
curl -L -o ./opencv-${OPENCV_VERSION}.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip
curl -L -o ./opencv_contrib-${OPENCV_VERSION}.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip
7z x opencv-${OPENCV_VERSION}.zip
7z x opencv_contrib-${OPENCV_VERSION}.zip

# cap_ffmpeg.cpp, cap_ffmpeg_impl.hppをパッチファイルに置き換える
export PATCH_DIR='/full_path_your_hogehoge/OpenCV_ffmpeg_patch'
cp ${PATCH_DIR}/cap_ffmpeg_for${OPENCV_VERSION//./}.cpp opencv-${OPENCV_VERSION}/modules/videoio/src/cap_ffmpeg.cpp
cp ${PATCH_DIR}/cap_ffmpeg_impl_for${OPENCV_VERSION//./}.hpp opencv-${OPENCV_VERSION}/modules/videoio/src/cap_ffmpeg_impl.hpp

cd opencv-${OPENCV_VERSION}
mkdir build
cd build

# cmakeの代わりに GUI版 cmakeでも、普通の ccmakeでも OK
# CUDA_ARCH_BIN, CUDA_ARCH_PTXは必要なバージョンを記述する
#   RTX 20xx系(Turing): 7.5
#   RTX 30xx系(Ampere): 8.6
#   (参考)Jetson Xavier NX(Volta): 7.2
export NVIDIA_ARCH='7.5'
cmake \
 -D WITH_OPENGL=ON \
 -D WITH_OPENMP=ON \
 -D WITH_TBB=ON \
 -D BUILD_opencv_world=ON \
 -D WITH_CUDA=ON \
 -D CUDA_ARCH_BIN=${NVIDIA_ARCH} \
 -D CUDA_ARCH_PTX=${NVIDIA_ARCH} \
 -D OPENCV_ENABLE_NONFREE:BOOL=ON \
 -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-${OPENCV_VERSION}/modules \
 -D WITH_GSTREAMER=ON \
 -D WITH_LIBV4L=ON \
 -D BUILD_opencv_python2=ON \
 -D BUILD_opencv_python3=ON \
 -D PYTHON_DEFAULT_EXECUTABLE=/usr/bin/python3 \
 -D BUILD_TESTS=OFF \
 -D BUILD_PERF_TESTS=OFF \
 -D BUILD_EXAMPLES=OFF \
 -D OPENCV_GENERATE_PKGCONFIG=ON \
 -D BUILD_PACKAGE=ON \
 -D CMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
 -D CMAKE_BUILD_TYPE=RELEASE \
 -D CMAKE_INSTALL_PREFIX=/usr/local \
 ..

sudo make -j$(nproc) install
```

もし、apt コマンドで /usr/lib, /usr/include 以下に別バージョンの OpenCV がインストールされていたら、
* その別バージョンが不要なら、apt remove で削除する。
* 必要なら、バージョン番号付きのライブラリファイル(libopencv\*.so.**\<version\>**)はそのまま残しておいて、それ以外を手動で削除する。  
削除するのは、  
バージョン番号なしのライブラリファイルたち(libopencv\*.so)、  
インクルードファイルたち(opencv2 ディレクトリまるごと)、  
pkgconfig ファイル(opencv.pc かな)
