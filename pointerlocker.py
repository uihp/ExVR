import time
import threading
import pygetwindow as gw
from pythonosc import udp_client
from pynput import mouse

import globals as g

OSC = udp_client.SimpleUDPClient('127.0.0.1', 39570)

class PointerLocker:
    def __init__(self):
        self.is_running = False
        self.x, self.y = 0, 0
        self.target_window = None
    def on_mouse_move(self, x, y):
        delta_x, delta_y = x - self.origin_x, y - self.origin_y
        self.x += delta_x
        self.y += delta_y
        g.raw.mouse.yaw = -(self.x / self.target_window.width * 75) % 360
        g.raw.mouse.pitch = -(self.y / self.target_window.height * 75) % 360
    def on_mouse_click(self, x, y, button, pressed):
        if button is not mouse.Button.left: return
        message = [1, 0, 0.0, int(pressed)]
        OSC.send_message("/VMT/Input/Trigger/Click", message)
    def thread_target(self):
        window_title = 'VRChat' if 1 else 'VR 视图'
        while self.is_running and self.target_window is None:
            if windows := gw.getWindowsWithTitle(window_title): self.target_window, = windows
            time.sleep(1 / 10)
        print('Target window found')
        mouse_controller = mouse.Controller()
        locked = False
        while self.is_running:
            if not locked and self.target_window.isActive:
                self.origin_x, self.origin_y = mouse_controller.position = self.target_window.center
                mouse_listener = mouse.Listener(on_click=self.on_mouse_click, on_move=self.on_mouse_move, suppress=True)
                mouse_listener.start()
                locked = True
                print('Pointer locked')
            if locked and not self.target_window.isActive:
                mouse_listener.stop()
                locked = False
                print('Pointer unlocked')
            time.sleep(1 / 20)
    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self.thread_target, daemon=True)
        self.thread.start()
    def stop(self):
        self.is_running = False
        self.thread.join()
