import sys
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
    QFrame,
    QCheckBox,
    QSlider,
    QMessageBox,
    QDialog,
    QScrollArea,
    QGridLayout
)
from PySide6.QtCore import QThread, Qt
from PySide6.QtGui import QDoubleValidator

import cv2
import winreg, shutil
from ctypes import windll

import utils.globals as g
from utils.actions import *
import utils.tracking
from utils.data import setup_data, save_data
from tracker.controller.controller import *
from tracker.face.face import draw_face_landmarks
from tracker.face.tongue import draw_tongue_position
from tracker.hand.hand import draw_hand_landmarks
from tracker.pose.pose import draw_pose_landmarks

class settings:
    camera_url = 'wss://192.168.31.150:5000/browsercam/signal'
    flip_x = False
    flip_y = False
    head_enable = True
    face_enable = True
    tougue_enable = True
    pose_enable = True
    hand_enable = True
    priority = 'ABOVE_NORMAL_PRIORITY_CLASS'
    @classmethod
    def toggle(cls, key): setattr(cls, key, not getattr(cls, key))

class VideoCaptureThread(QThread):
    def __init__(self):
        super().__init__()
        self.is_running = True
        self.tracker = utils.tracking.Tracker()
    def run(self):
        from rtcam import CameraThread
        self.camera = CameraThread(settings.camera_url)
        self.camera.start()
        while self.is_running:
            while self.camera.frame is None: time.sleep(0.1)
            rgb_image = self.camera.frame.to_ndarray(format='rgb24')
            if settings.flip_x: rgb_image = cv2.flip(rgb_image, 1)
            if settings.flip_y: rgb_image = cv2.flip(rgb_image, 0)
            self.tracker.process_frames(rgb_image)
            if g.config['Tracking']['Head']['enable'] or g.config['Tracking']['Face']['enable']:
                rgb_image = draw_face_landmarks(rgb_image)
            if g.config['Tracking']['Tongue']['enable']:
                rgb_image = draw_tongue_position(rgb_image)
            if g.config['Tracking']['Pose']['enable']:
                rgb_image = draw_pose_landmarks(rgb_image)
            if g.config['Tracking']['Hand']['enable']:
                rgb_image = draw_hand_landmarks(rgb_image)
            cv2.imshow('Camera View', cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        self.camera.stop()
    def stop(self):
        self.is_running = False
        self.tracker.stop()

class VideoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f'ExVR - Experience Virtual Reality')

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.steamvr_status_label = QLabel(self)
        layout.insertWidget(0, self.steamvr_status_label)

        flip_layout = QHBoxLayout()
        self.flip_x_checkbox = QCheckBox('Flip X', self)
        self.flip_x_checkbox.clicked.connect(lambda:settings.toggle('flip_x'))
        self.flip_x_checkbox.setChecked(settings.flip_x)
        flip_layout.addWidget(self.flip_x_checkbox)
        self.flip_y_checkbox = QCheckBox('Flip Y', self)
        self.flip_y_checkbox.clicked.connect(lambda:settings.toggle('flip_y'))
        self.flip_y_checkbox.setChecked(settings.flip_y)
        flip_layout.addWidget(self.flip_y_checkbox)
        layout.addLayout(flip_layout)

        self.camera_url_input = QLineEdit(self)
        self.camera_url_input.setPlaceholderText('Enter WebRTC camera URL')
        self.camera_url_input.textChanged.connect(lambda val:setattr(settings, 'camera_url', val))
        self.camera_url_input.setText(settings.camera_url)
        layout.addWidget(self.camera_url_input)

        self.priority_selection = QComboBox(self)
        self.priority_selection.addItems(['IDLE_PRIORITY_CLASS', 'BELOW_NORMAL_PRIORITY_CLASS', 'NORMAL_PRIORITY_CLASS', 'ABOVE_NORMAL_PRIORITY_CLASS', 'HIGH_PRIORITY_CLASS', 'REALTIME_PRIORITY_CLASS'])
        self.priority_selection.currentIndexChanged.connect(self.set_process_priority)
        layout.addWidget(self.priority_selection)
        self.priority_selection.setCurrentIndex(self.priority_selection.findText(settings.priority))

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

        self.toggle_button = QPushButton('Start Tracking', self)
        self.toggle_button.setStyleSheet('QPushButton { background-color: green; color: white; }')
        self.toggle_button.clicked.connect(self.toggle_camera)
        layout.addWidget(self.toggle_button)

        only_ingame_layout = QHBoxLayout()
        self.only_ingame_checkbox = QCheckBox('Only Ingame', self)
        self.only_ingame_checkbox.clicked.connect(lambda: self.toggle_only_in_game(self.only_ingame_checkbox.isChecked()))
        self.only_ingame_checkbox.setChecked(g.only_ingame)
        self.only_ingame_checkbox.setToolTip('Currently this only applies to hotkeys and mouse input and not head movement')
        self.only_ingame_game_input = QLineEdit(self)
        self.only_ingame_game_input.setPlaceholderText('window title / process name / VRChat, VRChat.exe, javaw.exe')
        self.only_ingame_game_input.textChanged.connect(lambda val:setattr(g, 'only_ingame_game', val))
        self.only_ingame_game_input.setText(g.only_ingame_game)
        only_ingame_layout.addWidget(self.only_ingame_checkbox)
        only_ingame_layout.addWidget(self.only_ingame_game_input)
        layout.addLayout(only_ingame_layout)

        separator_0 = QFrame(self)
        layout.addWidget(separator_0)

        reset_layout = QHBoxLayout()  # Create a QHBoxLayout for new reset buttons
        self.reset_head = QPushButton('Reset Head', self)
        self.reset_head.clicked.connect(reset_head)
        reset_layout.addWidget(self.reset_head)
        self.reset_eyes = QPushButton('Reset Eyes', self)
        self.reset_eyes.clicked.connect(reset_eye)
        reset_layout.addWidget(self.reset_eyes)
        self.reset_l_hand = QPushButton('Reset LHand', self)
        self.reset_l_hand.clicked.connect(lambda: reset_hand(True))
        reset_layout.addWidget(self.reset_l_hand)
        self.reset_r_hand = QPushButton('Reset RHand', self)
        self.reset_r_hand.clicked.connect(lambda: reset_hand(False))
        reset_layout.addWidget(self.reset_r_hand)
        layout.addLayout(reset_layout)

        checkbox_layout = QHBoxLayout()
        self.checkbox1 = QCheckBox('Head', self)
        self.checkbox1.clicked.connect(lambda: self.set_tracking_config('Head', self.checkbox1.isChecked()))
        checkbox_layout.addWidget(self.checkbox1)
        self.checkbox2 = QCheckBox('Face', self)
        self.checkbox2.clicked.connect(lambda: self.set_tracking_config('Face', self.checkbox2.isChecked()))
        checkbox_layout.addWidget(self.checkbox2)
        self.checkbox3 = QCheckBox('Tongue', self)
        self.checkbox3.clicked.connect(lambda: self.set_tracking_config('Tongue', self.checkbox3.isChecked()))
        checkbox_layout.addWidget(self.checkbox3)
        self.checkbox4 = QCheckBox('Hand', self)
        self.checkbox4.clicked.connect(lambda: self.set_tracking_config('Hand', self.checkbox4.isChecked()))
        checkbox_layout.addWidget(self.checkbox4)
        self.checkbox5 = QCheckBox('Pose', self)
        self.checkbox5.clicked.connect(lambda: self.set_tracking_config('Pose', self.checkbox5.isChecked()))
        checkbox_layout.addWidget(self.checkbox5)
        layout.addLayout(checkbox_layout)

        checkbox_layout_1 = QHBoxLayout()
        self.checkbox6 = QCheckBox('Hand Down', self)
        self.checkbox6.clicked.connect(lambda: self.toggle_hand_down(self.checkbox6.isChecked()))
        checkbox_layout_1.addWidget(self.checkbox6)
        self.checkbox7 = QCheckBox('Finger Action', self)
        self.checkbox7.clicked.connect(lambda: self.toggle_finger_action(self.checkbox7.isChecked()))
        checkbox_layout_1.addWidget(self.checkbox7)
        layout.addLayout(checkbox_layout_1)

        slider_layout = QHBoxLayout()
        self.slider1 = QSlider(Qt.Horizontal)
        self.slider2 = QSlider(Qt.Horizontal)
        self.slider3 = QSlider(Qt.Horizontal)
        self.slider1.setRange(1, 200)
        self.slider2.setRange(1, 200)
        self.slider3.setRange(1, 100)
        self.slider1.setSingleStep(1)
        self.slider2.setSingleStep(1)
        self.slider3.setSingleStep(1)
        self.label1 = QLabel(f'x {g.config['Tracking']['Hand']['x_scalar']:.2f}')
        self.label2 = QLabel(f'y {g.config['Tracking']['Hand']['y_scalar']:.2f}')
        self.label3 = QLabel(f'z {g.config['Tracking']['Hand']['z_scalar']:.2f}')
        self.slider1.valueChanged.connect(lambda value: self.set_scalar(value, 'x'))
        self.slider2.valueChanged.connect(lambda value: self.set_scalar(value, 'y'))
        self.slider3.valueChanged.connect(lambda value: self.set_scalar(value, 'z'))
        slider_layout.addWidget(self.label1)
        slider_layout.addWidget(self.slider1)
        slider_layout.addWidget(self.label2)
        slider_layout.addWidget(self.slider2)
        slider_layout.addWidget(self.label3)
        slider_layout.addWidget(self.slider3)
        layout.addLayout(slider_layout)

        separator_1 = QFrame(self)
        layout.addWidget(separator_1)

        controller_checkbox_layout = QHBoxLayout()
        self.controller_checkbox1 = QCheckBox('Left Controller', self)
        self.controller_checkbox1.clicked.connect(lambda: self.set_tracking_config('LeftController', self.controller_checkbox1.isChecked()))
        self.controller_checkbox2 = QCheckBox('Right Controller', self)
        self.controller_checkbox2.clicked.connect(lambda: self.set_tracking_config('RightController', self.controller_checkbox2.isChecked()))
        controller_checkbox_layout.addWidget(self.controller_checkbox1)
        controller_checkbox_layout.addWidget(self.controller_checkbox2)
        layout.addLayout(controller_checkbox_layout)

        controller_slider_layout = QHBoxLayout()
        self.controller_slider_x = QSlider(Qt.Horizontal)
        self.controller_slider_y = QSlider(Qt.Horizontal)
        self.controller_slider_z = QSlider(Qt.Horizontal)
        self.controller_slider_l = QSlider(Qt.Horizontal)
        self.controller_slider_x.setRange(-50, 50)
        self.controller_slider_y.setRange(-50, 50)
        self.controller_slider_z.setRange(-50, 50)
        self.controller_slider_l.setRange(0, 100)
        self.controller_slider_x.setSingleStep(1)
        self.controller_slider_y.setSingleStep(1)
        self.controller_slider_z.setSingleStep(1)
        self.controller_slider_l.setSingleStep(1)
        self.controller_label_x = QLabel(f'x {g.config['Tracking']['LeftController']['base_x']:.2f}')
        self.controller_label_y = QLabel(f'y {g.config['Tracking']['LeftController']['base_y']:.2f}')
        self.controller_label_z = QLabel(f'z {g.config['Tracking']['LeftController']['base_z']:.2f}')
        self.controller_label_l = QLabel(f'l {g.config['Tracking']['LeftController']['length']:.2f}')
        self.controller_slider_x.valueChanged.connect(lambda value: self.set_scalar(value, 'controller_x'))
        self.controller_slider_y.valueChanged.connect(lambda value: self.set_scalar(value, 'controller_y'))
        self.controller_slider_z.valueChanged.connect(lambda value: self.set_scalar(value, 'controller_z'))
        self.controller_slider_l.valueChanged.connect(lambda value: self.set_scalar(value, 'controller_l'))
        controller_slider_layout.addWidget(self.controller_label_x)
        controller_slider_layout.addWidget(self.controller_slider_x)
        controller_slider_layout.addWidget(self.controller_label_y)
        controller_slider_layout.addWidget(self.controller_slider_y)
        controller_slider_layout.addWidget(self.controller_label_z)
        controller_slider_layout.addWidget(self.controller_slider_z)
        controller_slider_layout.addWidget(self.controller_label_l)
        controller_slider_layout.addWidget(self.controller_slider_l)
        layout.addLayout(controller_slider_layout)

        separator_2 = QFrame(self)
        layout.addWidget(separator_2)

        mouse_layout = QHBoxLayout()
        self.mouse_checkbox = QCheckBox('Mouse', self)
        self.mouse_checkbox.clicked.connect(lambda: self.toggle_mouse(self.mouse_checkbox.isChecked()))
        self.mouse_checkbox.setChecked(g.config['Mouse']['enable'])
        self.mouse_slider_x = QSlider(Qt.Horizontal)
        self.mouse_slider_y = QSlider(Qt.Horizontal)
        self.mouse_slider_dx = QSlider(Qt.Horizontal)
        self.mouse_slider_x.setRange(0, 360)
        self.mouse_slider_y.setRange(0, 360)
        self.mouse_slider_dx.setRange(0, 20)
        self.mouse_slider_x.setSingleStep(1)
        self.mouse_slider_y.setSingleStep(1)
        self.mouse_slider_dx.setSingleStep(1)
        self.mouse_label_x = QLabel(f'x {int(g.config['Mouse']['scalar_x']*100)}')
        self.mouse_label_y = QLabel(f'y {int(g.config['Mouse']['scalar_y']*100)}')
        self.mouse_label_dx = QLabel(f'dx {g.config['Mouse']['dx']:.2f}')
        self.mouse_slider_x.valueChanged.connect(lambda value: self.set_scalar(value, 'mouse_x'))
        self.mouse_slider_y.valueChanged.connect(lambda value: self.set_scalar(value, 'mouse_y'))
        self.mouse_slider_dx.valueChanged.connect(lambda value: self.set_scalar(value, 'mouse_dx'))
        mouse_layout.addWidget(self.mouse_checkbox)
        mouse_layout.addWidget(self.mouse_label_x)
        mouse_layout.addWidget(self.mouse_slider_x)
        mouse_layout.addWidget(self.mouse_label_y)
        mouse_layout.addWidget(self.mouse_slider_y)
        mouse_layout.addWidget(self.mouse_label_dx)
        mouse_layout.addWidget(self.mouse_slider_dx)
        layout.addLayout(mouse_layout)

        separator_3 = QFrame(self)
        layout.addWidget(separator_3)

        config_layout = QHBoxLayout()
        self.set_face_button = QPushButton('Set Face', self)
        self.set_face_button.clicked.connect(self.face_dialog)
        config_layout.addWidget(self.set_face_button)
        self.update_config_button = QPushButton('Update Config', self)
        self.update_config_button.clicked.connect(lambda:(g.update_configs(),self.update_checkboxes(), self.update_sliders()))
        config_layout.addWidget(self.update_config_button)
        self.save_config_button = QPushButton('Save Config', self)
        self.save_config_button.clicked.connect(g.save_configs)
        config_layout.addWidget(self.save_config_button)
        layout.addLayout(config_layout)
        self.update_checkboxes()
        self.update_sliders()
        self.video_thread = None
        self.controller_thread = None
    def save_data(self):
        data=deepcopy(g.default_data)
        for i, (key, edits) in enumerate(self.lineEdits.items()):
            idx=i+1
            v = float(edits[0].text())
            s = float(edits[1].text())
            w = float(edits[2].text())
            max_value = float(edits[3].text())
            e = self.checkBoxes[key].isChecked()
            data['BlendShapes'][idx]['v'] = v
            data['BlendShapes'][idx]['s'] = s
            data['BlendShapes'][idx]['w'] = w
            data['BlendShapes'][idx]['max'] = max_value
            data['BlendShapes'][idx]['e'] = e
        save_data(data)
        self.dialog.close()
    def face_dialog(self):
        self.dialog = QDialog(self)
        self.dialog.setWindowTitle('Face Setting')
        self.dialog.resize(self.width, self.height)  # Set a fixed size for the dialog
        layout = QVBoxLayout(self.dialog)
        # Create a scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        # Create a widget to hold the form layout
        form_widget = QWidget()
        form_layout = QGridLayout(form_widget)  # Use a grid layout for better alignment
        # Add header labels for the form
        form_layout.addWidget(QLabel('BlendShape'), 0, 0)
        form_layout.addWidget(QLabel('Value'), 0, 1)
        form_layout.addWidget(QLabel('Shifting'), 0, 2)
        form_layout.addWidget(QLabel('Weight'), 0, 3)
        form_layout.addWidget(QLabel('Max'), 0, 4)
        form_layout.addWidget(QLabel('Enabled'), 0, 5)
        # Store QLineEdit and QCheckBox references
        self.lineEdits = {}
        self.checkBoxes = {}
        # Create input fields for each blend shape
        double_validator = QDoubleValidator()
        blendshape_data,_=setup_data()
        for i, blendshape in enumerate(blendshape_data['BlendShapes'][1:], start=1):  # Start from row 1
            key = blendshape['k']
            v_edit = QLineEdit(str(round(blendshape['v'],2)))
            v_edit.setValidator(double_validator)
            s_edit = QLineEdit(str(round(blendshape['s'],2)))
            s_edit.setValidator(double_validator)
            w_edit = QLineEdit(str(round(blendshape['w'],2)))
            w_edit.setValidator(double_validator)
            max_edit = QLineEdit(str(round(blendshape['max'],2)))
            max_edit.setValidator(double_validator)
            e_check = QCheckBox()
            e_check.setChecked(blendshape['e'])
            # Save references to the QLineEdit and QCheckBox
            self.lineEdits[key] = (v_edit, s_edit, w_edit, max_edit)
            self.checkBoxes[key] = e_check
            # Add widgets to the grid layout
            form_layout.addWidget(QLabel(key), i, 0)  # Blend shape key in column 0
            form_layout.addWidget(v_edit, i, 1)  # v_edit in column 1
            form_layout.addWidget(s_edit, i, 2)  # s_edit in column 2
            form_layout.addWidget(w_edit, i, 3)  # w_edit in column 3
            form_layout.addWidget(max_edit, i, 4)  # max_edit in column 4
            form_layout.addWidget(e_check, i, 5)  # e_check in column 5
        # Add the form layout to the scroll area
        scroll_area.setWidget(form_widget)
        layout.addWidget(scroll_area)  # Add the scroll area to the dialog layout
        # Add a Save Config button
        self.save_config_button = QPushButton('Save Config', self.dialog)
        self.save_config_button.clicked.connect(self.save_data)
        layout.addWidget(self.save_config_button)
        self.dialog.exec_()
    def update_checkboxes(self):
        self.flip_x_checkbox.setChecked(settings.flip_x)
        self.flip_y_checkbox.setChecked(settings.flip_y)
        self.checkbox1.setChecked(g.config['Tracking']['Head']['enable'])
        self.checkbox2.setChecked(g.config['Tracking']['Face']['enable'])
        self.checkbox3.setChecked(g.config['Tracking']['Tongue']['enable'])
        self.checkbox4.setChecked(g.config['Tracking']['Hand']['enable'])
        self.checkbox5.setChecked(g.config['Tracking']['Pose']['enable'])
        self.checkbox6.setChecked(g.config['Tracking']['Hand']['enable_hand_down'])
        self.checkbox7.setChecked(g.config['Tracking']['Hand']['enable_finger_action'])
        self.controller_checkbox1.setChecked(g.config['Tracking']['LeftController']['enable'])
        self.controller_checkbox2.setChecked(g.config['Tracking']['RightController']['enable'])
        self.mouse_checkbox.setChecked(g.config['Mouse']['enable'])
    def set_scalar(self, value, axis):
        slider_value = value / 100.0
        if axis == 'x':
            g.config['Tracking']['Hand']['x_scalar'] = slider_value
            self.label1.setText(f'x {slider_value:.2f}')
        elif axis == 'y':
            g.config['Tracking']['Hand']['y_scalar'] = slider_value
            self.label2.setText(f'y {slider_value:.2f}')
        elif axis == 'z':
            g.config['Tracking']['Hand']['z_scalar'] = slider_value
            self.label3.setText(f'z {slider_value:.2f}')
        elif axis == 'controller_x':
            g.config['Tracking']['LeftController']['base_x'] = slider_value
            g.config['Tracking']['RightController']['base_x'] = -slider_value
            self.controller_label_x.setText(f'x {slider_value:.2f}')
        elif axis == 'controller_y':
            g.config['Tracking']['LeftController']['base_y'] = slider_value
            g.config['Tracking']['RightController']['base_y'] = slider_value
            self.controller_label_y.setText(f'y {slider_value:.2f}')
        elif axis == 'controller_z':
            g.config['Tracking']['LeftController']['base_z'] = slider_value
            g.config['Tracking']['RightController']['base_z'] = slider_value
            self.controller_label_z.setText(f'z {slider_value:.2f}')
        elif axis == 'controller_l':
            g.config['Tracking']['LeftController']['length'] = slider_value
            g.config['Tracking']['RightController']['length'] = slider_value
            self.controller_label_l.setText(f'l {slider_value:.2f}')
        elif axis == 'mouse_x':
            g.config['Mouse']['scalar_x']=slider_value*100
            self.mouse_label_x.setText(f'x {int(slider_value*100)}')
        elif axis == 'mouse_y':
            g.config['Mouse']['scalar_y']=slider_value*100
            self.mouse_label_y.setText(f'y {int(slider_value*100)}')
        elif axis == 'mouse_dx':
            g.config['Mouse']['dx']=slider_value
            self.mouse_label_dx.setText(f'dx {slider_value:.2f}')
    def update_sliders(self):
        x_scalar = g.config['Tracking']['Hand']['x_scalar']
        y_scalar = g.config['Tracking']['Hand']['y_scalar']
        z_scalar = g.config['Tracking']['Hand']['z_scalar']
        self.slider1.setValue(int(x_scalar * 100))
        self.slider2.setValue(int(y_scalar * 100))
        self.slider3.setValue(int(z_scalar * 100))
        self.label1.setText(f'x {x_scalar:.2f}')
        self.label2.setText(f'y {y_scalar:.2f}')
        self.label3.setText(f'z {z_scalar:.2f}')
        controller_x = g.config['Tracking']['LeftController']['base_x']
        controller_y = g.config['Tracking']['LeftController']['base_y']
        controller_z = g.config['Tracking']['LeftController']['base_z']
        controller_l = g.config['Tracking']['LeftController']['length']
        self.controller_slider_x.setValue(int(controller_x * 100))
        self.controller_slider_y.setValue(int(controller_y * 100))
        self.controller_slider_z.setValue(int(controller_z * 100))
        self.controller_slider_l.setValue(int(controller_l * 100))
        self.controller_label_x.setText(f'x {controller_x:.2f}')
        self.controller_label_y.setText(f'y {controller_y:.2f}')
        self.controller_label_z.setText(f'z {controller_z:.2f}')
        self.controller_label_l.setText(f'l {controller_l:.2f}')
        mouse_x=g.config['Mouse']['scalar_x']
        mouse_y=g.config['Mouse']['scalar_y']
        mouse_dx=g.config['Mouse']['dx']
        self.mouse_slider_x.setValue(int(mouse_x))
        self.mouse_slider_y.setValue(int(mouse_y))
        self.mouse_slider_dx.setValue(int(mouse_dx * 100))
        self.mouse_label_x.setText(f'x {int(mouse_x)}')
        self.mouse_label_y.setText(f'y {int(mouse_y)}')
        self.mouse_label_dx.setText(f'dx {mouse_dx:.2f}')
    def reset_hotkeys(self):
        stop_hotkeys()
        apply_hotkeys()
        if self.video_thread is None:
            stop_hotkeys()
    def set_tracking_config(self, key, value):
        if key in g.config['Tracking']:
            g.config['Tracking'][key]['enable'] = value
        if key == 'LeftController':
            g.controller.left_hand.force_enable = value
        if key == 'RightController':
            g.controller.right_hand.force_enable = value
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
            self.display_message('Error','Invalid priority index')
            return False
        priority_class = priority_classes[priority_key]
        current_pid = os.getpid()  # Get the current process ID
        handle = windll.kernel32.OpenProcess(0x0200 | 0x0400, False, current_pid)  # Open the current process
        success = windll.kernel32.SetPriorityClass(handle, priority_class)
        windll.kernel32.CloseHandle(handle)
        print('Finished setting priority')
        settings.priority = priority_key
    def display_message(self,title,message,style=''):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setText(message)
        msg_box.setWindowTitle(title)
        msg_box.setStyleSheet(style)
        msg_box.exec_()
        return
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
        self.update_checkboxes()
        self.update_sliders()
        if self.video_thread and self.video_thread.isRunning():
            self.toggle_button.setText('Start Tracking')
            self.toggle_button.setStyleSheet('QPushButton { background-color: green; color: white; }')
            self.thread_stopped()
        else:
            self.toggle_button.setText('Stop Tracking')
            self.toggle_button.setStyleSheet('QPushButton { background-color: red; color: white; }')
            self.video_thread = VideoCaptureThread()
            self.video_thread.start()
            # controller
            self.controller_thread = ControllerApp()
            self.controller_thread.start()
    def thread_stopped(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()
            self.video_thread = None
        if self.controller_thread:
            self.controller_thread.stop()
            self.controller_thread.wait()
            self.controller_thread = None
    def closeEvent(self, event):
        self.thread_stopped()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.show()
    sys.exit(app.exec())
