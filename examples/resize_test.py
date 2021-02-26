#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__author__  = "hy"
__version__ = "1.00"
__date__    = "29 Aug 2020"

import time
import cv2
import numpy as np
import jetson.utils

# OUT_SIZE = [[640, 360], [720, 405], [800, 450], [864, 486], [1008, 567], [1024, 576], [1152, 648], [1280, 720], [1296, 729], [1440, 810], [1600, 900], [1920, 1080]]
OUT_SIZE = [[640, 360], [1024, 576], [1280, 720], [1600, 900], [1920, 1080], [3840, 2160]]
# OUT_SIZE = [[1920, 1080]]
# OUT_SIZE = [[160, 90]]
# OUT_SIZE = [[640, 360], [1024, 576], [1280, 720]]

IN_SIZE = [[1024, 576]]
# IN_SIZE = [[1920, 1080]]
# IN_SIZE = [[2560, 1440]]

# ITP_MODE = cv2.INTER_NEAREST
ITP_MODE = cv2.INTER_LINEAR
# ITP_MODE = cv2.INTER_CUBIC
# ITP_MODE = cv2.INTER_AREA
# ITP_MODE = cv2.INTER_LANCZOS4

# ITP_MODE_JETSON_UTILS = jetson.utils.INTER_NEAREST
ITP_MODE_JETSON_UTILS = jetson.utils.INTER_LINEAR
# ITP_MODE_JETSON_UTILS = jetson.utils.INTER_CUBIC
# ITP_MODE_JETSON_UTILS = jetson.utils.INTER_AREA
# ITP_MODE_JETSON_UTILS = jetson.utils.INTER_LANCZOS4
# ITP_MODE_JETSON_UTILS = jetson.utils.INTER_SPLINE36

class HogeHogeResize(object):
    def __init__(self):
        self.src_gpu = cv2.cuda_GpuMat()
        self.dst_gpu = cv2.cuda_GpuMat()

    def Go(self, src, width, height, device='cpu'):
        dsize = (width, height)
        if device == 'cpu':
            # CPU version.
            dst = cv2.resize(src, dsize, interpolation = ITP_MODE)
        else:
            # GPU version.
            self.src_gpu.upload(src)
            self.dst_gpu = cv2.cuda.resize(self.src_gpu, dsize, interpolation = ITP_MODE)
            dst = self.dst_gpu.download()

        return dst

    def Go2(self, src, dst):
        # jetson-utils version.
        jetson.utils.cudaResize(src, dst, ITP_MODE_JETSON_UTILS)
        jetson.utils.cudaDeviceSynchronize()

def test_resize():
    # 画像を取得
    # cpu, gpu. (BGR)
    img = cv2.imread("src.jpg")
    # jetson-utils. (RGBA)
    img2 = jetson.utils.loadImage('src.jpg', format='rgba8')

    # 3ch(BGR) -> 4ch(BGRA).
    img_a = np.full((img.shape[0], img.shape[1]), 255).astype(np.uint8)
    img_4ch = np.zeros((img.shape[0], img.shape[1], 4)).astype(np.uint8)
    img_4ch[...,0:3] = img
    img_4ch[...,3]   = img_a

    # input size.
    in_X, in_Y = IN_SIZE[0]
    # cpu, gpu.
    img_in = cv2.resize(img_4ch, (in_X, in_Y))
    # jetson-utils.
    img_in2 = jetson.utils.cudaAllocMapped(width=in_X, height=in_Y, format=img2.format)
    jetson.utils.cudaResize(img2, img_in2, jetson.utils.INTER_LINEAR)

    hoge_resize = HogeHogeResize()
    if ITP_MODE != cv2.INTER_LANCZOS4:
        dummy_img_scaled = hoge_resize.Go(img_in, in_X, in_Y, 'gpu')   # 時間計測のため。CUDAの初期化時間をキャンセル。

    loop_cnt = 1000
    if ITP_MODE != cv2.INTER_LANCZOS4:
        device_list = ['cpu', 'gpu', 'jetson-utils']
    else:
        device_list = ['cpu', 'jetson-utils']
    # device_list = ['jetson-utils']

    print(in_X, in_Y, "-->")
    for X, Y in OUT_SIZE:
        print(X, Y)

        # resize()を計測.
        for d in device_list:   # CPU, CUDA, jetson-utils.
            if d != 'jetson-utils':
                img_IN = img_in
                start = time.time()
                for i in range(loop_cnt):
                    img_OUT = hoge_resize.Go(img_IN, X, Y, d)
                end_time = time.time() - start
            else:
                img_IN = img_in2
                img_OUT = jetson.utils.cudaAllocMapped(width=X, height=Y, format=img_IN.format)
                start = time.time()
                for i in range(loop_cnt):
                    hoge_resize.Go2(img_IN, img_OUT)
                end_time = time.time() - start

            if d == 'cpu':
                print('resize(CPU) end_time [msec],',end_time*1000/loop_cnt)
                cv2.imwrite("dst_resize_cpu.jpg", img_OUT)
            elif d == 'gpu':
                print('resize(CUDA) end_time [msec],',end_time*1000/loop_cnt)
                cv2.imwrite("dst_resize_gpu.jpg", img_OUT)
            else:
                print('resize(jetson-utils) end_time [msec],',end_time*1000/loop_cnt)
                jetson.utils.saveImage("dst_resize_jetson-utils.jpg", img_OUT)



if __name__ == '__main__':
    test_resize()
