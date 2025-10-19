import os, sys
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QLineEdit,
    QComboBox,
    QHBoxLayout,
    QCheckBox
)
from PySide6.QtCore import QThread

import time
import cv2
import winreg, shutil
from ctypes import windll

import globals as g
from tracking import Tracker

class VideoCaptureThread(QThread):
    def __init__(self, tracker):
        super().__init__()
        self.is_running = True
        self.tracker = tracker
    def run(self):
        from rtcam import CameraThread
        self.camera = CameraThread(g.settings.camera_url)
        self.camera.start()
        while self.is_running:
            if self.camera.frame is None:
                time.sleep(0.1)
                continue
            image_rgb = self.camera.frame.to_ndarray(format='rgb24')
            if g.settings.flip_x: image_rgb = cv2.flip(image_rgb, 1)
            if g.settings.flip_y: image_rgb = cv2.flip(image_rgb, 0)
            image_rgb = self.tracker.process_frames(image_rgb)
            cv2.imshow('Camera View', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        self.camera.stop()
    def stop(self):
        self.is_running = False

class VideoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle(f'ExVR - Experience Virtual Reality')

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.steamvr_status_label = QLabel(self)
        layout.insertWidget(0, self.steamvr_status_label)

        self.install_state, steamvr_driver_path, vrcfacetracking_path, check_steamvr_path = self.install_checking()
        if check_steamvr_path is not None:
            self.steamvr_status_label.setText('SteamVR Installed')
            self.steamvr_status_label.setStyleSheet('color: green; font-weight: bold;')
        else:
            self.steamvr_status_label.setText('SteamVR Not Installed')
            self.steamvr_status_label.setStyleSheet('color: red; font-weight: bold;')
        if self.install_state:
            self.install_button = QPushButton('Uninstall Drivers', self)
            self.install_button.setStyleSheet('')
        else:
            self.install_button = QPushButton('Install Drivers', self)
            self.install_button.setStyleSheet('QPushButton { background-color: blue; color: white; }')
        self.install_button.clicked.connect(self.install_function)
        layout.addWidget(self.install_button)

        self.camera_url_input = QLineEdit(self)
        self.camera_url_input.setPlaceholderText('Enter WebRTC camera URL')
        self.camera_url_input.textChanged.connect(lambda val:setattr(g.settings, 'camera_url', val))
        self.camera_url_input.setText(g.settings.camera_url)
        layout.addWidget(self.camera_url_input)

        flip_layout = QHBoxLayout()
        self.flip_x_checkbox = QCheckBox('Flip X', self)
        self.flip_x_checkbox.clicked.connect(lambda:g.settings.toggle('flip_x'))
        self.flip_x_checkbox.setChecked(g.settings.flip_x)
        flip_layout.addWidget(self.flip_x_checkbox)
        self.flip_y_checkbox = QCheckBox('Flip Y', self)
        self.flip_y_checkbox.clicked.connect(lambda:g.settings.toggle('flip_y'))
        self.flip_y_checkbox.setChecked(g.settings.flip_y)
        flip_layout.addWidget(self.flip_y_checkbox)
        layout.addLayout(flip_layout)

        only_ingame_layout = QHBoxLayout()
        self.only_ingame_checkbox = QCheckBox('Only Ingame', self)
        self.only_ingame_checkbox.clicked.connect(lambda: self.setattr(g.settings, 'only_ingame', self.only_ingame_checkbox.isChecked()))
        self.only_ingame_checkbox.setChecked(g.settings.only_ingame)
        self.only_ingame_checkbox.setToolTip('Currently this only applies to hotkeys and mouse input and not head movement')
        self.only_ingame_game_input = QLineEdit(self)
        self.only_ingame_game_input.setPlaceholderText('window title / process name / VRChat, VRChat.exe, javaw.exe')
        self.only_ingame_game_input.textChanged.connect(lambda val:setattr(g.settings, 'only_ingame_game', val))
        self.only_ingame_game_input.setText(g.settings.only_ingame_game)
        only_ingame_layout.addWidget(self.only_ingame_checkbox)
        only_ingame_layout.addWidget(self.only_ingame_game_input)
        layout.addLayout(only_ingame_layout)

        self.priority_selection = QComboBox(self)
        self.priority_selection.addItems(['IDLE_PRIORITY_CLASS', 'BELOW_NORMAL_PRIORITY_CLASS', 'NORMAL_PRIORITY_CLASS', 'ABOVE_NORMAL_PRIORITY_CLASS', 'HIGH_PRIORITY_CLASS', 'REALTIME_PRIORITY_CLASS'])
        self.priority_selection.currentIndexChanged.connect(self.set_process_priority)
        layout.addWidget(self.priority_selection)
        self.priority_selection.setCurrentIndex(self.priority_selection.findText(g.settings.priority))

        self.toggle_button = QPushButton('Start Tracking', self)
        self.toggle_button.setStyleSheet('QPushButton { background-color: green; color: white; }')
        self.toggle_button.clicked.connect(self.toggle_camera)
        layout.addWidget(self.toggle_button)

        self.tracker = Tracker()

        self.video_thread = None
        self.toggle_camera()
    def set_tracking_config(self, key, value):
        if key in g.config['Tracking']: g.config['Tracking'][key]['enable'] = value
    def install_checking(self):
        # Open registry key to get Steam installation path
        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r'SOFTWARE\WOW6432Node\Valve\Steam',
                0, winreg.KEY_READ
            ) as reg_key:
                steam_path, _ = winreg.QueryValueEx(reg_key, 'InstallPath')
            steamvr_driver_path = os.path.join(steam_path, 'steamapps', 'common', 'SteamVR', 'drivers')
            check_steamvr_path = os.path.join(steam_path, 'steamapps', 'common', 'SteamVR', 'bin')
            if not os.path.exists(check_steamvr_path):
                check_steamvr_path = None
            vrcfacetracking_path = os.path.join(os.getenv('APPDATA'), 'VRCFaceTracking', 'CustomLibs')
            vrcfacetracking_module_path = os.path.join(vrcfacetracking_path, 'VRCFT-MediapipePro.dll')
            # Check all required paths
            required_paths = [vrcfacetracking_module_path] + [
                os.path.join(steamvr_driver_path, driver)
                for driver in ['vrto3d']
            ]
            if all(os.path.exists(path) for path in required_paths):
                return True, steamvr_driver_path, vrcfacetracking_path, check_steamvr_path
            else:
                return False, steamvr_driver_path, vrcfacetracking_path, check_steamvr_path
        except Exception as e:
            print(f'Error accessing registry or file system: {e}')
            return False, None, None, None
    def set_process_priority(self):
        priority_key = self.priority_selection.currentText()
        print(priority_key)
        # Define a mapping of priority indexes to their corresponding priority classes
        priority_classes = {
            'IDLE_PRIORITY_CLASS': 0x00000040,
            'BELOW_NORMAL_PRIORITY_CLASS': 0x00004000,
            'NORMAL_PRIORITY_CLASS': 0x00000020,  # NORMAL_PRIORITY_CLASS
            'ABOVE_NORMAL_PRIORITY_CLASS': 0x00008000,
            'HIGH_PRIORITY_CLASS': 0x00000080,
            'REALTIME_PRIORITY_CLASS': 0x00000100
        }
        # Check if the index is valid
        if priority_key not in priority_classes:
            print('Error', 'Invalid priority index')
            return False
        priority_class = priority_classes[priority_key]
        current_pid = os.getpid()  # Get the current process ID
        handle = windll.kernel32.OpenProcess(0x0200 | 0x0400, False, current_pid)  # Open the current process
        success = windll.kernel32.SetPriorityClass(handle, priority_class)
        windll.kernel32.CloseHandle(handle)
        print('Finished setting priority')
        g.settings.priority = priority_key
    def install_function(self):
        self.install_state, steamvr_driver_path, vrcfacetracking_path, check_steamvr_path = self.install_checking()
        if check_steamvr_path is not None:
            self.steamvr_status_label.setText('SteamVR Installed')
            self.steamvr_status_label.setStyleSheet('color: green; font-weight: bold;')
        else:
            self.steamvr_status_label.setText('SteamVR Not Installed')
            self.steamvr_status_label.setStyleSheet('color: red; font-weight: bold;')
        if self.install_state:
            # Uninstall process
            dll_path = os.path.join(vrcfacetracking_path, 'VRCFT-MediapipePro.dll')
            error_occurred = False
            drivers_to_remove = ['vrto3d']
            for driver in drivers_to_remove:
                dir_path = os.path.join(steamvr_driver_path, driver)
                try:
                    shutil.rmtree(dir_path)
                except FileNotFoundError:
                    pass
                except Exception as e:
                    error_occurred = True
                if os.path.exists(dir_path):
                    error_occurred = True
            if error_occurred:
                self.display_message('Error', 'SteamVR is running, Please close SteamVR and try again.')
                return
            try:
                os.remove(dll_path)
            except PermissionError:
                self.display_message('Error', 'VRCFT is running, please close VRCFT and try again.')
                return
            self.install_button.setText('Install Drivers')
            self.install_button.setStyleSheet('QPushButton { background-color: blue; color: white; }')
        else:
            # Install process
            for driver in ['vrto3d']:
                source = os.path.join('./drivers', driver)
                destination = os.path.join(steamvr_driver_path, driver)
                if not os.path.exists(destination):
                    shutil.copytree(source, destination)
            dll_source = os.path.join('./drivers', 'VRCFT-MediapipePro.dll')
            dll_destination = os.path.join(
                vrcfacetracking_path, 'VRCFT-MediapipePro.dll'
            )
            if not os.path.exists(dll_destination):
                os.makedirs(os.path.dirname(dll_destination), exist_ok=True)
                shutil.copy(dll_source, dll_destination)
            self.install_button.setText('Uninstall Drivers')
            self.install_button.setStyleSheet('')
    def toggle_camera(self):
        if self.video_thread and self.video_thread.isRunning():
            self.toggle_button.setText('Start Tracking')
            self.toggle_button.setStyleSheet('QPushButton { background-color: green; color: white; }')
            self.thread_stopped()
        else:
            self.toggle_button.setText('Stop Tracking')
            self.toggle_button.setStyleSheet('QPushButton { background-color: red; color: white; }')
            self.video_thread = VideoCaptureThread(self.tracker)
            self.tracker.start()
            self.video_thread.start()
    def thread_stopped(self):
        if self.video_thread:
            self.tracker.stop()
            self.video_thread.stop()
            self.video_thread.wait()
            self.video_thread = None
    def closeEvent(self, event):
        self.thread_stopped()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.show()
    sys.exit(app.exec())
