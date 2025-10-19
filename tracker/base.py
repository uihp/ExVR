from abc import abstractmethod
from threading import Thread
from time import sleep

class BackendBase:
    def __init__(self): pass
    @abstractmethod
    def update(self): pass

class TrackerBase:
    backend: BackendBase
    def __init__(self):
        self.sending = False
        self.refresh_rate = 1 / 60
    def __send_thread(self):
        while self.sending:
            self.backend.update()
            sleep(self.refresh_rate)
    def start_sending(self):
        self.thread = Thread(target=self.__send_thread, daemon=True)
        self.thread.start()
    def stop_sending(self):
        self.sending = False
        self.thread.join()
