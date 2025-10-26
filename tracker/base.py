from abc import abstractmethod
from threading import Thread
from time import sleep, perf_counter

class BackendBase:
    def __init__(self): pass
    @abstractmethod
    def update(self): ...

class TrackerBase:
    backend: BackendBase
    listeners = []
    def __init__(self):
        self.is_running = False
        self.refresh_rate = 1 / 60
        self.smoothing_freq = 5 / 1000
    @abstractmethod
    def smooth_frame(self, dt): ...
    def __smoothing_thread(self):
        last = perf_counter()
        while self.is_running:
            now = perf_counter()
            self.smooth_frame(now - last)
            last = now
            sleep(self.smoothing_freq)
    def __sending_thread(self):
        while self.is_running:
            self.backend.update()
            sleep(self.refresh_rate)
    def start(self):
        self.is_running = True
        self.smoothing_thread = Thread(target=self.__smoothing_thread, daemon=True)
        self.smoothing_thread.start()
        self.sending_thread = Thread(target=self.__sending_thread, daemon=True)
        self.sending_thread.start()
        for listener in self.listeners: listener.start()
    def stop(self):
        self.is_running = False
        self.smoothing_thread.join()
        self.sending_thread.join()
        for listener in self.listeners: listener.stop()
