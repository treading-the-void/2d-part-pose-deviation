"""USB RGB camera capture thread."""
import time, queue
import cv2
from PyQt5.QtCore import QThread, pyqtSignal
from .settings import CameraConfig

class UsbCameraThread(QThread):
    frame_ready = pyqtSignal(object)  # 把帧发到主线程

    def __init__(self, cfg: CameraConfig):
        super().__init__()
        self.cfg = cfg
        self.frame_q = queue.Queue(maxsize=cfg.queue_max)
        self._running = False
        self._cam = None

    def _set_params(self, cap: cv2.VideoCapture):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        # Keep the RGB camera's original MJPG/manual-exposure setup.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        cap.set(cv2.CAP_PROP_EXPOSURE, self.cfg.exposure)
        cap.set(cv2.CAP_PROP_GAIN, self.cfg.gain)
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def start_grab(self):
        if self._running:
            return True
        self._cam = cv2.VideoCapture(self.cfg.video_id, self.cfg.backend)
        if not self._cam.isOpened():
            return False
        self._set_params(self._cam)
        self._running = True
        self.start()
        return True

    def stop_grab(self):
        self._running = False
        self.wait()
        if self._cam:
            self._cam.release()
        self._clear_queue()

    def get_frame(self, timeout=0.1):
        try:
            return self.frame_q.get(timeout=timeout)
        except queue.Empty:
            return None

    def run(self):
        retry = 0
        while self._running:
            if not self._cam or not self._cam.isOpened():
                if retry < self.cfg.retry_times:
                    retry += 1
                    time.sleep(self.cfg.retry_interval)
                    self._cam = cv2.VideoCapture(self.cfg.video_id, self.cfg.backend)
                    if self._cam.isOpened():
                        self._set_params(self._cam)
                    continue
                else:
                    self._running = False
                    break

            ret, frame = self._cam.read()
            if not ret:
                retry += 1
                time.sleep(self.cfg.thread_sleep)
                continue
            retry = 0

            if not self.frame_q.full():
                self.frame_q.put(frame)
                self.frame_ready.emit(frame)  # 实时发信号（可注释，用定时器拉取也行）

            time.sleep(self.cfg.thread_sleep)

    def _clear_queue(self):
        while not self.frame_q.empty():
            self.frame_q.get_nowait()
