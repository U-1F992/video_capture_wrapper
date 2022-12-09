import multiprocessing
from time import perf_counter

import cv2

from video_capture_wrapper import VideoCaptureWrapper


def main():
    multiprocessing.freeze_support()

    video_capture = VideoCaptureWrapper(0)
    # video_capture = cv2.VideoCapture(0)

    try:
        while True:
            start_time = perf_counter()
            _, mat = video_capture.read()
            print(perf_counter() - start_time)

            cv2.imshow("window", mat)
            cv2.waitKey(1)

    except KeyboardInterrupt:
        pass

    video_capture.release()


if __name__ == "__main__":
    main()
