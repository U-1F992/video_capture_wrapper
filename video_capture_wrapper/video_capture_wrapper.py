from __future__ import annotations

import ctypes
import multiprocessing
import multiprocessing.sharedctypes
import multiprocessing.synchronize
import signal
from typing import cast

import cv2
import numpy as np


def _update(args: tuple, props: dict[int, float], buffer: ctypes.Array[ctypes.c_uint8], ready: multiprocessing.synchronize.Event, cancel: multiprocessing.synchronize.Event):
    """
    画像を取得してbufferを更新する

    [参考](https://qiita.com/kakinaguru_zo/items/eda129635816ad871e9d#%E4%B8%A6%E5%88%97%E5%87%A6%E7%90%86%E3%81%AB%E3%82%88%E3%82%8B%E7%94%BB%E9%9D%A2%E6%9B%B4%E6%96%B0%E3%81%AE%E9%AB%98%E9%80%9F%E5%8C%96)
    """

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    video_capture = cv2.VideoCapture(*args)
    if not video_capture.isOpened():
        raise IOError()

    _set_props(video_capture, props)

    try:
        while not cancel.is_set():
            ret, mat = cast("tuple[bool, cv2.Mat]", video_capture.read())
            if not ret:
                continue

            ready.clear()
            memoryview(buffer).cast('B')[:] = memoryview(mat).cast('B')[:]
            ready.set()

    finally:
        video_capture.release()


def _set_props(video_capture: cv2.VideoCapture, props: dict[int, float]):
    for key, value in props.items():
        try:
            video_capture.set(key, value)
        except:
            pass


def _get_props(video_capture: cv2.VideoCapture) -> dict[int, float]:
    ids = [cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_WIDTH]
    return cast("dict[int, float]", dict([[prop, video_capture.get(prop)] for prop in ids]))


def _get_information(args: tuple, in_props: dict[int, float]) -> tuple[tuple[int, int, int], dict[int, float]]:
    
    video_capture = cv2.VideoCapture(*args)
    if not video_capture.isOpened():
        raise IOError()

    # 入力されたプロパティを設定してから、現在のプロパティ一覧を取得する
    _set_props(video_capture, in_props)
    out_props = _get_props(video_capture)

    try:
        ret, mat = cast("tuple[bool, cv2.Mat]", video_capture.read())
        if not ret:
            raise IOError()

        return mat.shape, out_props

    finally:
        video_capture.release()


class VideoCaptureWrapper:

    def __init__(self, *args) -> None:
        self.__args = ()
        self.__shape = cast(int, 0), cast(int, 0), cast(int, 0)
        self.__props: dict[int, float] = {}

        self.__buffer = multiprocessing.sharedctypes.RawArray(
            ctypes.c_uint8, 1)
        self.__ready = multiprocessing.Event()
        self.__cancel = multiprocessing.Event()
        self.__enqueue = multiprocessing.Process()

        self.__released = cast(bool, True)

        if len(args) == 0:
            return

        self.open(*args)

    def open(self, *args):
        if not self.__released:
            raise RuntimeError()

        self.__args = args
        self.__shape, self.__props = _get_information(
            self.__args, self.__props)

        height, width, channels = self.__shape
        self.__buffer = multiprocessing.sharedctypes.RawArray(
            ctypes.c_uint8, height * width * channels)

        self.__ready = multiprocessing.Event()
        self.__cancel = multiprocessing.Event()
        self.__enqueue = multiprocessing.Process(target=_update, args=(
            self.__args, self.__props, self.__buffer, self.__ready, self.__cancel), daemon=True)
        self.__enqueue.start()

        self.__released = cast(bool, False)

    def get(self, propId: int):
        if self.__released:
            raise RuntimeError()

        return self.__props[propId]

    def set(self, propId: int, value: float):
        if self.__released:
            raise RuntimeError()

        self.__props[propId] = value
        self.release()
        self.open(*self.__args)

        return cast(bool, True)

    def read(self):
        if self.__released:
            raise RuntimeError()

        self.__ready.wait()
        return cast(bool, True), np.reshape(self.__buffer, self.__shape).copy()

    def isOpened(self):
        return not self.__released

    def release(self):
        if self.__released:
            return

        self.__cancel.set()
        self.__enqueue.join()
        self.__released = True

    def __del__(self):
        try:
            self.release()
        except:
            pass

if __name__ == "__main__":
    multiprocessing.freeze_support()
    