#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__author__  = "hy"
__version__ = "1.00"
__date__    = "8 Sep 2020"

import time
import cv2
import numpy as np
import jetson.utils

# IMG_SIZE = [[640, 360], [720, 405], [800, 450], [864, 486], [1008, 567], [1024, 576], [1152, 648], [1280, 720], [1296, 729], [1440, 810], [1600, 900], [1920, 1080]]
# IMG_SIZE = [[640, 360], [1024, 576], [1280, 720], [1600, 900], [1920, 1080], [3840, 2160]]
# IMG_SIZE = [[640, 360]]
IMG_SIZE = [[1920, 1080]]
# IMG_SIZE = [[160, 90]]

class HogeHogeSplit(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.src_gpu = cv2.cuda_GpuMat()
        self.dst_gpu = [cv2.cuda_GpuMat() for i in range(4)]

        self.color_cpu = np.zeros((height, width, 3), dtype=np.uint8)
        self.alpha_cpu = np.zeros((height, width, 1), dtype=np.uint8)
        self.color_gpu = cv2.cuda_GpuMat(height, width, cv2.CV_8UC3)
        self.alpha_gpu = cv2.cuda_GpuMat(height, width, cv2.CV_8UC1)

    def Go(self, src, device='cpu'):
        ch = src.shape[2]
        if device == 'cpu':
            # CPU version.
            dst = cv2.split(src)
        elif device == 'numpy':
            # NumPy version.
            dst = [src[:,:,i] for i in range(ch)]
        else:
            # GPU version.
            self.src_gpu.upload(src)
            self.dst_gpu = cv2.cuda.split(self.src_gpu)
            dst = [self.dst_gpu[i].download() for i in range(ch)]

        return dst

    def Go_31(self, src, device='cpu'):
        if device == 'cpu':
            # CPU version.
            dst = cv2.mixChannels([src], [self.color_cpu, self.alpha_cpu], (0,0, 1,1, 2,2, 3, 3))
        elif device == 'numpy':
            # NumPy version.
            dst = [src[...,0:3], src[...,3]]
        else:
            # GPU version.
            # print("GPU version not support mixChannels.")
            dst = [self.color_gpu, self.alpha_gpu]

        return dst

    def Go2(self, src, dst):
        # jetson-utils version.
        jetson.utils.cudaSplit(src, dst)
        jetson.utils.cudaDeviceSynchronize()

    def Go2_31(self, src, dst_color, dst_alpha):
        # jetson-utils version.
        jetson.utils.cudaSplit(src, dst_color, dst_alpha)
        jetson.utils.cudaDeviceSynchronize()



class HogeHogeMerge(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.src_gpu = [cv2.cuda_GpuMat() for i in range(4)]
        self.dst_gpu = cv2.cuda_GpuMat()

        self.merged_cpu = np.zeros((height, width, 4), dtype=np.uint8)
        self.merged_gpu = cv2.cuda_GpuMat(height, width, cv2.CV_8UC4)

    def Go(self, src, device='cpu'):
        ch = len(src)
        if device == 'cpu':
            # CPU version.
            dst = cv2.merge(src)
        elif device == 'numpy':
            # NumPy version.
            for i in range(ch):
                self.merged_cpu[...,i] = src[i]
            dst = self.merged_cpu
        else:
            # GPU version.
            for i in range(ch):
                self.src_gpu[i].upload(src[i])
            self.dst_gpu = cv2.cuda.merge(self.src_gpu)
            dst = self.dst_gpu

        return dst

    def Go31(self, src_color, src_alpha, device='cpu'):
        if device == 'cpu':
            # CPU version.
            dst = cv2.mixChannels([src_color, src_alpha], [self.merged_cpu], (0,0, 1,1, 2,2, 3,3))[0]
        elif device == 'numpy':
            # NumPy version.
            self.merged_cpu[...,0:3] = src_color
            self.merged_cpu[...,3] = src_alpha
            dst = self.merged_cpu
        else:
            # GPU version.
            # print("GPU version not support mixChannels.")
            dst = self.merged_gpu

        return dst

    def Go2(self, src, dst):
        # jetson-utils version.
        jetson.utils.cudaMerge(src, dst)
        jetson.utils.cudaDeviceSynchronize()

    def Go2_31(self, src_color, dst, src_alpha):
        # jetson-utils version.
        jetson.utils.cudaMerge(src_color, dst, src_alpha)
        jetson.utils.cudaDeviceSynchronize()



def test_split_merge():
    # 画像を取得
    # cpu, gpu. (BGR)
    img = cv2.imread("src.jpg")
    # jetson-utils. (RGBA)
    img2 = jetson.utils.loadImage('src.jpg', format='rgba8')

    # 3ch(BGR) -> 4ch(BGRA).
    img_a = np.full((img.shape[0], img.shape[1]), 255, dtype = np.uint8)
    img_4ch = np.zeros((img.shape[0], img.shape[1], 4), dtype = np.uint8)
    img_4ch[...,0:3] = img
    img_4ch[...,3]   = img_a

    # input size.
    in_X, in_Y = IMG_SIZE[0]
    # cpu, gpu.
    img_in = cv2.resize(img_4ch, (in_X, in_Y))
    # jetson-utils.
    img_in2 = jetson.utils.cudaAllocMapped(width=in_X, height=in_Y, format=img2.format)
    jetson.utils.cudaResize(img2, img_in2, jetson.utils.INTER_LINEAR)

    hoge_split = HogeHogeSplit(in_X, in_Y)
    dummy_img_split = hoge_split.Go(img_in, 'gpu')   # 時間計測のため。CUDAの初期化時間をキャンセル。

    hoge_merge = HogeHogeMerge(in_X, in_Y)
    dummy_img_out = hoge_merge.Go(dummy_img_split, 'gpu')   # 時間計測のため。CUDAの初期化時間をキャンセル。

    loop_cnt = 10000
    device_list = ['cpu', 'gpu', 'numpy', 'jetson-utils']
    # device_list = ['jetson-utils']

    # split(), merge()を計測.
    for d in device_list:   # CPU, CUDA, NumPy, jetson-utils.
        # split.
        if d != 'jetson-utils':
            img_IN = img_in
            start = time.time()
            for i in range(loop_cnt):
                img_SPLIT = hoge_split.Go(img_IN, d)
                # img_SPLIT = hoge_split.Go(img_IN, 'numpy')
            end_time = time.time() - start
        else:
            img_IN = img_in2
            # print(img_IN)
            X = img_IN.width
            Y = img_IN.height
            fmt = 'gray8'
            ch = img_IN.channels
            img_SPLIT = [jetson.utils.cudaAllocMapped(width=X, height=Y, format=fmt) for i in range(ch)]
            # print("img_SPLIT len:", len(img_SPLIT))
            # print(img_SPLIT)
            # for i in range(len(img_SPLIT)):
            #     print(img_SPLIT[i])
            start = time.time()
            for i in range(loop_cnt):
                hoge_split.Go2(img_IN, img_SPLIT)
            end_time = time.time() - start

        if d == 'cpu':
            print('split(CPU) end_time [msec],',end_time*1000/loop_cnt)
        elif d == 'gpu':
            print('split(CUDA) end_time [msec],',end_time*1000/loop_cnt)
        elif d == 'numpy':
            print('split(NumPy) end_time [msec],',end_time*1000/loop_cnt)
        else:
            print('split(jetson-utils) end_time [msec],',end_time*1000/loop_cnt)
            # jetson.utils.saveImage("dst_split_jetson-utils.jpg", img_SPLIT[0])



        # merge.
        if d != 'jetson-utils':
            start = time.time()
            for i in range(loop_cnt):
                img_OUT = hoge_merge.Go(img_SPLIT, d)
                # img_OUT = hoge_merge.Go(img_SPLIT, 'cpu')
            end_time = time.time() - start
        else:
            X = img_SPLIT[0].width
            Y = img_SPLIT[0].height
            fmt = 'rgba8' if len(img_SPLIT) == 4 else 'rgb8'
            img_OUT = jetson.utils.cudaAllocMapped(width=X, height=Y, format=fmt)
            # print(img_OUT)
            start = time.time()
            for i in range(loop_cnt):
                hoge_merge.Go2(img_SPLIT, img_OUT)
            end_time = time.time() - start

        if d == 'cpu':
            print('merge(CPU) end_time [msec],',end_time*1000/loop_cnt)
            cv2.imwrite("dst_merge_cpu.jpg", img_OUT)
        elif d == 'gpu':
            print('merge(CUDA) end_time [msec],',end_time*1000/loop_cnt)
            cv2.imwrite("dst_merge_gpu.jpg", img_OUT)
        elif d == 'numpy':
            print('merge(NumPy) end_time [msec],',end_time*1000/loop_cnt)
            cv2.imwrite("dst_merge_numpy.jpg", img_OUT)
        else:
            print('merge(jetson-utils) end_time [msec],',end_time*1000/loop_cnt)
            jetson.utils.saveImage("dst_merge_jetson-utils.jpg", img_OUT)



        # split. 31
        if d != 'jetson-utils':
            img_IN = img_in
            start = time.time()
            for i in range(loop_cnt):
                img_COLOR, img_ALPHA = hoge_split.Go_31(img_IN, d)
            # print(img_COLOR.shape, img_ALPHA.shape)
            end_time = time.time() - start
        else:
            img_IN = img_in2
            # print(img_IN)
            X = img_IN.width
            Y = img_IN.height
            img_COLOR = jetson.utils.cudaAllocMapped(width=X, height=Y, format='rgb8')
            img_ALPHA = jetson.utils.cudaAllocMapped(width=X, height=Y, format='gray8')
            # print(img_COLOR)
            # print(img_ALPHA)
            start = time.time()
            for i in range(loop_cnt):
                hoge_split.Go2_31(img_IN, img_COLOR, img_ALPHA)
            end_time = time.time() - start

        if d == 'cpu':
            print('split31(CPU) end_time [msec],',end_time*1000/loop_cnt)
            # cv2.imwrite("dst_split31_cpu_c.jpg", img_COLOR)
            # cv2.imwrite("dst_split31_cpu_a.jpg", img_ALPHA)
        elif d == 'gpu':
            print('split31(CUDA) end_time [msec],',end_time*1000/loop_cnt)
        elif d == 'numpy':
            print('split31(NumPy) end_time [msec],',end_time*1000/loop_cnt)
            # cv2.imwrite("dst_split31_numpy_c.jpg", img_COLOR)
            # cv2.imwrite("dst_split31_numpy_a.jpg", img_ALPHA)
        else:
            print('split31(jetson-utils) end_time [msec],',end_time*1000/loop_cnt)
            # jetson.utils.saveImage("dst_split31_jetson-utils_c.jpg", img_COLOR)
            # jetson.utils.saveImage("dst_split31_jetson-utils_a.jpg", img_ALPHA)



        # merge. 31
        if d != 'jetson-utils':
            start = time.time()
            for i in range(loop_cnt):
                img_OUT = hoge_merge.Go31(img_COLOR, img_ALPHA, d)
            end_time = time.time() - start
        else:
            X = img_COLOR.width
            Y = img_COLOR.height
            img_OUT = jetson.utils.cudaAllocMapped(width=X, height=Y, format='rgba8')
            # print(img_OUT)
            start = time.time()
            for i in range(loop_cnt):
                hoge_merge.Go2_31(img_COLOR, img_OUT, img_ALPHA)
            end_time = time.time() - start

        if d == 'cpu':
            print('merge31(CPU) end_time [msec],',end_time*1000/loop_cnt)
            cv2.imwrite("dst_merge31_cpu.jpg", img_OUT)
        elif d == 'gpu':
            print('merge31(CUDA) end_time [msec],',end_time*1000/loop_cnt)
        elif d == 'numpy':
            print('merge31(NumPy) end_time [msec],',end_time*1000/loop_cnt)
            cv2.imwrite("dst_merge31_numpy.jpg", img_OUT)
        else:
            print('merge31(jetson-utils) end_time [msec],',end_time*1000/loop_cnt)
            jetson.utils.saveImage("dst_merge31_jetson-utils.jpg", img_OUT)



if __name__ == '__main__':
    test_split_merge()
