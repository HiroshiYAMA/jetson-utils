#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__author__  = "hy"
__version__ = "1.00"
__date__    = "1 Sep 2020"

import time
import cv2
import numpy as np
import jetson.utils

# OUT_SIZE = [[640, 360], [720, 405], [800, 450], [864, 486], [1008, 567], [1024, 576], [1152, 648], [1280, 720], [1296, 729], [1440, 810], [1600, 900], [1920, 1080]]
# OUT_SIZE = [[640, 360], [1024, 576], [1280, 720], [1600, 900], [1920, 1080], [3840, 2160]]
# OUT_SIZE = [[1600, 900]]

# IN_SIZE = [[1024, 576]]
IN_SIZE = [[1920, 1080]]
# IN_SIZE = [[2560, 1440]]

# COLOR_MODE_IN = cv2.COLOR_BGR2BGRA
# COLOR_MODE_OUT = cv2.COLOR_BGRA2RGBA
#
COLOR_MODE_IN = cv2.COLOR_BGR2YUV_I420
COLOR_MODE_OUT = cv2.COLOR_YUV2RGBA_I420
# COLOR_MODE_OUT = cv2.COLOR_YUV2BGRA_I420
#
# COLOR_MODE_IN = cv2.COLOR_BGR2YUV
# COLOR_MODE_OUT = cv2.COLOR_YUV2RGB

# COLOR_MODE_IN_JETSON_UTILS = "bgra8"
# COLOR_MODE_OUT_JETSON_UTILS = "rgba8"
#
COLOR_MODE_IN_JETSON_UTILS = "i420"
COLOR_MODE_OUT_JETSON_UTILS = "rgba8"
# COLOR_MODE_OUT_JETSON_UTILS = "bgra8"

class HogeHogeConvertColor(object):
    def __init__(self):
        self.src_gpu = cv2.cuda_GpuMat()
        self.dst_gpu = cv2.cuda_GpuMat()

    def Go(self, src, color_format, device='cpu'):
        if device == 'cpu':
            # CPU version.
            dst = cv2.cvtColor(src, color_format)
        else:
            # GPU version.
            self.src_gpu.upload(src)
            self.dst_gpu = cv2.cuda.cvtColor(self.src_gpu, color_format)
            dst = self.dst_gpu.download()

        return dst

    def Go2(self, src, dst):
        # jetson-utils version.
        jetson.utils.cudaConvertColor(src, dst)
        jetson.utils.cudaDeviceSynchronize()

def test_color_conv():
    # 画像を取得
    # cpu, gpu. (BGR)
    img_tmp = cv2.imread("src.jpg")
    # jetson-utils. (RGBA)
    img2_tmp = jetson.utils.loadImage('src.jpg', format='rgba8')

    # input size.
    in_X, in_Y = IN_SIZE[0]
    # cpu, gpu.
    img = cv2.resize(img_tmp, (in_X, in_Y))
    # jetson-utils.
    img2 = jetson.utils.cudaAllocMapped(width=in_X, height=in_Y, format=img2_tmp.format)
    jetson.utils.cudaResize(img2_tmp, img2, jetson.utils.INTER_LINEAR)

    # 3ch(BGR) -> 4ch(BGRA) or 3ch(YUV. same as I420).
    img_in = cv2.cvtColor(img, COLOR_MODE_IN)

    # (RGBA -> BGRA or I420)
    img_in2 = jetson.utils.cudaAllocMapped(width=img2.width, height=img2.height, format=COLOR_MODE_IN_JETSON_UTILS)
    jetson.utils.cudaConvertColor(img2, img_in2)
    jetson.utils.cudaDeviceSynchronize()
    del img2

    hoge_color_conv = HogeHogeConvertColor()
    # dummy_img_color_conv = hoge_color_conv.Go(img_in, COLOR_MODE_OUT, 'gpu')   # 時間計測のため。CUDAの初期化時間をキャンセル。

    loop_cnt = 10000
    # device_list = ['cpu', 'gpu', 'jetson-utils']
    device_list = ['cpu', 'jetson-utils']   # 'gpu' is not support I420.

    # convert colorを計測.
    print(COLOR_MODE_IN_JETSON_UTILS, "->", COLOR_MODE_OUT_JETSON_UTILS)
    for d in device_list:   # CPU, CUDA, jetson-utils.
        if d != 'jetson-utils':
            img_IN = img_in
            start = time.time()
            for i in range(loop_cnt):
                img_OUT = hoge_color_conv.Go(img_IN, COLOR_MODE_OUT, d)
            end_time = time.time() - start
        else:
            img_IN = img_in2
            img_OUT = jetson.utils.cudaAllocMapped(width=img_IN.width, height=img_IN.height, format=COLOR_MODE_OUT_JETSON_UTILS)
            start = time.time()
            for i in range(loop_cnt):
                hoge_color_conv.Go2(img_IN, img_OUT)
            end_time = time.time() - start

        if d == 'cpu':
            print('convert color(CPU) end_time [msec],',end_time*1000/loop_cnt)
            cv2.imwrite("dst_cpu.jpg", img_OUT)
        elif d == 'gpu':
            print('convert color(CUDA) end_time [msec],',end_time*1000/loop_cnt)
            cv2.imwrite("dst_gpu.jpg", img_OUT)
        else:
            print('convert color(jetson-utils) end_time [msec],',end_time*1000/loop_cnt)
            jetson.utils.saveImage("dst_jetson-utils.jpg", img_OUT)



if __name__ == '__main__':
    test_color_conv()
