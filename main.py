import sys
import os
import cv2
import json
import re

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QSlider, QFrame, QComboBox,
    QMessageBox, QDialog, QListWidget, QInputDialog, QToolButton, QMenu, QAction
)
from PyQt5.QtCore import QTimer, Qt, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage, QPalette, QColor, QIcon

from pygrabber.dshow_graph import FilterGraph
from ultralytics import YOLO

from config import CFG
from camera_thread import CameraThread


class VideoPlayerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.label = QLabel("Video here", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: #ffffff; font-size: 16px;")
        self.enabled_objects = []
        self.enabled_alerts = []
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None
        self.is_paused = False
        self.total_frames = 0
        self.current_frame = 0
        self.model = YOLO(CFG.WEIGHTS)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.detection_interval = 15
        self.last_detection_result = None

    def start_camera(self, camera_index=0):
        self.stop_video()
        os.environ["OPENCV_VIDEOIO_DEBUG"] = "1"
        os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
        self.cap = cv2.VideoCapture(camera_index)
        if self.cap.isOpened():
            self.total_frames = 0
            self.current_frame = 0
            self.is_paused = False
            self.timer.start(100)
        else:
            print("Could not open camera")

    def open_video(self, video_path):
        self.stop_video()
        try:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise IOError(f"Error opening video: {video_path}")
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame = 0
            self.is_paused = False
            self.timer.start(30)
        except Exception as e:
            print(f"Error opening video: {e}")

    def pause_video(self):
        self.is_paused = not self.is_paused

    def stop_video(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.cap = None
        self.is_paused = False
        self.current_frame = 0
        self.total_frames = 0

    def seek_frame(self, frame_num):
        if self.cap and self.cap.isOpened() and frame_num < self.total_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            self.current_frame = frame_num

    def update_detections_config(self, enabled_objects):
        self.enabled_objects = enabled_objects
        print(f"Updated object list: {self.enabled_objects}")

    def get_detections_config(self):
        return self.enabled_objects

    def update_alerts_config(self, enabled_alerts):
        self.enabled_alerts = enabled_alerts
        print(f"Updated alerts list: {self.enabled_alerts}")

    def get_alerts_config(self):
        return self.enabled_alerts

    def update_frame(self):
        if not self.is_paused and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                self.stop_video()
                return

            self.current_frame += 1

            # Виконуємо детекцію кожні `detection_interval` кадрів
            if (self.current_frame % self.detection_interval == 0) or (self.last_detection_result is None):
                results = list(self.model(frame, CFG.CONFIDENCE))
                self.last_detection_result = results
            else:
                results = self.last_detection_result

            alert_triggered = False

            # Якщо є детекція на кадрі, накладаємо рамки та підписи
            if len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    cls_id = int(box.cls[0])  # Ідентифікатор класу
                    conf = float(box.conf[0])  # Впевненість детекції
                    x1, y1, x2, y2 = box.xyxy[0]
                    label = self.model.names[cls_id]

                    # Фільтруємо об'єкти залежно від порогу conf
                    if conf < 0.5:
                        continue

                    # Накладання рамок і підписів
                    if label in self.enabled_objects:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness=2)
                        cv2.putText(
                            frame, f"{label} {conf:.2f}",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1
                        )

                    # Тривога (якщо клас у списку сигналізацій)
                    if label in self.enabled_alerts:
                        alert_triggered = True

            # Якщо є тривога, виділяємо межі червоним
            if alert_triggered:
                h, w, _ = frame.shape
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), thickness=20)

            # Конвертуємо кадр для відображення
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled = pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label.setPixmap(scaled)

    def get_current_position(self):
        return self.current_frame

    def get_total_frames(self):
        return self.total_frames


def _get_camera_list():
    graph = FilterGraph()
    devices = graph.get_input_devices()
    camera_list = []
    for i, device_name in enumerate(devices):
        camera_list.append((device_name, i))
    return camera_list


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config_file = "config.json"
        if not os.path.exists(self.config_file):
            default_config = {
                "cameras": [],
                "phones": []
            }
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
        with open(self.config_file, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.setWindowIcon(QIcon("icon.png"))
        self.setWindowTitle("Military Equipment Detection System")
        self.resize(1200, 700)
        self._apply_custom_theme()
        self.vehicle_classes = [
            "apc", "army-truck", "bmp", "bus", "car", "imv",
            "missile", "mt-lb", "person", "rocket", "rocket-artillery", "tank"
        ]
        self.vehicle_actions = {}
        self.alert_actions = {}
        self.camera_threads = {}
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        self.video_player = VideoPlayerWidget(self)
        self.camera_combo = QComboBox()
        self.camera_combo.currentIndexChanged.connect(self.on_camera_combo_changed)
        self._populate_main_camera_combo()
        self.btn_add_camera = QPushButton("Add camera")
        self.btn_add_camera.clicked.connect(self.add_camera_clicked)
        self.btn_remove_camera = QPushButton("Remove camera")
        self.btn_remove_camera.clicked.connect(self.remove_camera_clicked)
        self.btn_open_camera = QPushButton("Show camera")
        self.btn_open_camera.clicked.connect(self.open_camera_clicked)
        self.btn_stop_demo = QPushButton("Stop")
        self.btn_stop_demo.clicked.connect(self.stop_video_clicked)
        self.btn_open_video = QPushButton("Open video")
        self.btn_open_video.clicked.connect(self.open_video_clicked)
        self.btn_play_pause = QPushButton("Pause/Resume")
        self.btn_play_pause.clicked.connect(self.video_player.pause_video)
        self.btn_stop = QPushButton("Stop/Close")
        self.btn_stop.clicked.connect(self.stop_video_clicked)
        self.btn_phones = QPushButton("Phones")
        self.btn_phones.clicked.connect(self.show_phones_dialog)
        self.btn_objects_menu = QToolButton()
        self.btn_objects_menu.setText("Select classes")
        self.btn_objects_menu.setPopupMode(QToolButton.InstantPopup)
        self.objects_menu = QMenu(self)
        self.btn_objects_menu.setMenu(self.objects_menu)
        self.btn_alerts_menu = QToolButton()
        self.btn_alerts_menu.setText("Select alerts")
        self.btn_alerts_menu.setPopupMode(QToolButton.InstantPopup)
        self.alerts_menu = QMenu(self)
        self.btn_alerts_menu.setMenu(self.alerts_menu)
        self._create_vehicle_actions()
        self._create_alert_actions()
        self.slider_video = QSlider(Qt.Horizontal)
        self.slider_video.setMinimum(0)
        self.slider_video.valueChanged.connect(self.on_slider_value_changed)
        self.slider_update_timer = QTimer()
        self.slider_update_timer.timeout.connect(self.update_slider)
        self.slider_update_timer.start(100)
        left_panel = QVBoxLayout()
        lbl_select_camera = QLabel("Select camera:", self)
        left_panel.addWidget(lbl_select_camera)
        left_panel.addWidget(self.camera_combo)
        left_panel.addWidget(self.btn_add_camera)
        left_panel.addWidget(self.btn_remove_camera)
        left_panel.addWidget(self.btn_open_camera)
        left_panel.addWidget(self.btn_stop_demo)
        left_panel.addWidget(self.btn_phones)
        left_panel.addStretch(1)
        right_panel = QVBoxLayout()
        right_panel.addWidget(self.btn_open_video)
        right_panel.addWidget(self.btn_play_pause)
        right_panel.addWidget(self.btn_stop)
        right_panel.addWidget(self.btn_objects_menu)
        right_panel.addWidget(self.btn_alerts_menu)
        right_panel.addStretch(1)
        top_layout = QHBoxLayout()
        top_layout.addLayout(left_panel)
        top_layout.addWidget(self.video_player, stretch=3)
        top_layout.addLayout(right_panel)
        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.slider_video)
        main_widget.setLayout(main_layout)
        self._auto_start_cameras_from_config()

    def _auto_start_cameras_from_config(self):
        for camera_obj in self.config.get("cameras", []):
            cam_name = camera_obj["name"]
            cam_id = camera_obj["id"]
            if cam_id not in self.camera_threads:
                alerts_dict = camera_obj.get("alerts", {})
                alerts_for_new_camera = [k for k, v in alerts_dict.items() if v is True]
                phones_list = self.config.get("phones", [])
                worker = CameraThread(cam_id, alerts_for_new_camera, phones_list)
                worker.camera_event.connect(self.on_camera_event)
                self.camera_threads[cam_id] = worker
                worker.start()
                print(f"[MainWindow] Auto-start camera thread: {cam_name} (ID={cam_id})")

    def on_camera_event(self, cam_id, frame):
        pass

    def closeEvent(self, event):
        self.video_player.stop_video()
        super().closeEvent(event)

    def _create_vehicle_actions(self):
        self.objects_menu.clear()
        self.vehicle_actions.clear()
        for cls_name in self.vehicle_classes:
            action = QAction(cls_name, self, checkable=True)
            action.setChecked(False)
            action.triggered.connect(self.on_vehicle_action_triggered)
            self.vehicle_actions[cls_name] = action
            self.objects_menu.addAction(action)

    def _create_alert_actions(self):
        self.alerts_menu.clear()
        self.alert_actions.clear()
        for cls_name in self.vehicle_classes:
            action = QAction(cls_name, self, checkable=True)
            action.setChecked(False)
            action.triggered.connect(self.on_alert_action_triggered)
            self.alert_actions[cls_name] = action
            self.alerts_menu.addAction(action)

    @pyqtSlot(bool)
    def on_vehicle_action_triggered(self, checked):
        self.save_vehicle_alert_config()
        QTimer.singleShot(0, lambda: self.btn_objects_menu.showMenu())

    @pyqtSlot(bool)
    def on_alert_action_triggered(self, checked):
        self.save_vehicle_alert_config()
        QTimer.singleShot(0, lambda: self.btn_alerts_menu.showMenu())

    def get_checked_vehicle_actions(self):
        return [name for name, act in self.vehicle_actions.items() if act.isChecked()]

    def get_checked_alert_actions(self):
        return [name for name, act in self.alert_actions.items() if act.isChecked()]

    def on_camera_combo_changed(self, index: int):
        camera_id = self.camera_combo.itemData(index)
        camera = self.getCamera(camera_id)
        if camera is not None:
            detected_objects = camera.get("detected_objects", {})
            alerts = camera.get("alerts", {})
            for cls_name, act in self.vehicle_actions.items():
                act.blockSignals(True)
                act.setChecked(detected_objects.get(cls_name, False))
                act.blockSignals(False)
            for cls_name, act in self.alert_actions.items():
                act.blockSignals(True)
                act.setChecked(alerts.get(cls_name, False))
                act.blockSignals(False)

    def add_camera_clicked(self):
        available_all = _get_camera_list()
        existing_ids = set()
        for i in range(self.camera_combo.count()):
            existing_ids.add(self.camera_combo.itemData(i))
        not_added = [(name, cid) for (name, cid) in available_all if cid not in existing_ids]
        if not not_added:
            QMessageBox.information(self, "Cameras", "All found cameras are already added.")
            return
        dlg = CameraAddDialog(not_added, self)
        if dlg.exec_() == QDialog.Accepted:
            selected_name, selected_id = dlg.get_selected_camera()
            if self.camera_combo.count() == 1 and self.camera_combo.itemData(0) == -1:
                self.camera_combo.clear()
            self.camera_combo.addItem(selected_name, selected_id)
            self.camera_combo.setEnabled(True)
            new_cam = {
                "name": selected_name,
                "id": selected_id,
                "detected_objects": {},
                "alerts": {}
            }
            self.config["cameras"].append(new_cam)
            self.save_config()
            print(f"Added camera: {selected_name} (ID={selected_id})")
            self.camera_combo.setCurrentIndex(self.camera_combo.count() - 1)
            if selected_id not in self.camera_threads:
                phones_list = self.config.get("phones", [])
                worker = CameraThread(selected_id, [], phones_list, self)
                worker.camera_event.connect(self.on_camera_event)
                self.camera_threads[selected_id] = worker
                worker.start()
        else:
            print("Canceled adding camera")

    def remove_camera_clicked(self):
        idx = self.camera_combo.currentIndex()
        if idx >= 0:
            text = self.camera_combo.itemText(idx)
            camera_id = self.camera_combo.itemData(idx)
            if camera_id != -1:
                if camera_id in self.camera_threads:
                    worker = self.camera_threads[camera_id]
                    worker.stop()
                    worker.wait()
                    del self.camera_threads[camera_id]
                self.camera_combo.removeItem(idx)
                if "cameras" in self.config:
                    self.config["cameras"] = [
                        c for c in self.config["cameras"] if not (c["id"] == camera_id and c["name"] == text)
                    ]
                self.save_config()
                if self.camera_combo.count() == 0:
                    self.camera_combo.addItem("No cameras found", -1)
                    self.camera_combo.setEnabled(False)
                print(f"Removed camera: {text}")
            else:
                QMessageBox.warning(self, "Warning", "Cannot remove 'No cameras found'")
        else:
            QMessageBox.warning(self, "Warning", "Camera list is empty")

    def open_camera_clicked(self):
        camera_index = self.camera_combo.currentData()
        if camera_index is not None and camera_index != -1:
            enabled_objects = self.get_checked_vehicle_actions()
            enabled_alerts = self.get_checked_alert_actions()
            self.video_player.update_detections_config(enabled_objects)
            self.video_player.update_alerts_config(enabled_alerts)
            self.video_player.start_camera(camera_index)
            self.set_video_controls_visible(False)
        else:
            QMessageBox.warning(self, "Warning", "No available cameras to open!")

    def open_video_clicked(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open video", "", "Video Files (*.mp4 *.avi *.mkv)")
        if file_name:
            enabled_objs = self.get_checked_vehicle_actions()
            print(f"Removed camera: {enabled_objs}")
            enabled_alrs = self.get_checked_alert_actions()
            self.video_player.update_detections_config(enabled_objs)
            self.video_player.update_alerts_config(enabled_alrs)
            self.video_player.open_video(file_name)
            self.set_video_controls_visible(True)

    def stop_video_clicked(self):
        self.video_player.stop_video()
        self.set_video_controls_visible(False)
        camera_id = self.camera_combo.currentData()
        if camera_id in self.camera_threads:
            worker = self.camera_threads[camera_id]
            worker.stop()
            worker.wait()
            enabled_alerts = self.get_checked_alert_actions()
            phones_list = self.config.get("phones", [])
            new_worker = CameraThread(camera_id, enabled_alerts, phones_list, self)
            new_worker.camera_event.connect(self.on_camera_event)
            self.camera_threads[camera_id] = new_worker
            new_worker.start()

    def on_slider_value_changed(self, value):
        if self.video_player.cap and self.video_player.cap.isOpened():
            total_frames = self.video_player.get_total_frames()
            if total_frames > 0:
                self.video_player.seek_frame(value)

    def update_slider(self):
        if self.video_player.cap and self.video_player.cap.isOpened():
            total_frames = self.video_player.get_total_frames()
            if total_frames > 0:
                current_pos = self.video_player.get_current_position()
                self.slider_video.setRange(0, total_frames - 1)
                if not self.slider_video.isSliderDown():
                    self.slider_video.setValue(current_pos)
            else:
                self.slider_video.setRange(0, 0)

    def show_phones_dialog(self):
        phones_list = self.config.get("phones", [])
        dialog = PhoneManagerDialog(phones_list, self)
        if dialog.exec_() == QDialog.Accepted:
            updated_phones = dialog.get_phones()
            self.config["phones"] = updated_phones
            self.save_config()
            print("Updated phone list:", updated_phones)
        else:
            print("Canceled phone changes")

    def set_video_controls_visible(self, visible: bool):
        self.btn_play_pause.setVisible(visible)
        self.btn_stop.setVisible(visible)
        self.slider_video.setVisible(visible)

    def getCamera(self, camera_id):
        if camera_id == -1:
            return None
        for cam in self.config.get("cameras", []):
            if cam["id"] == camera_id:
                return cam
        return None

    def save_vehicle_alert_config(self):
        camera_id = self.camera_combo.currentData()
        cam = self.getCamera(camera_id)
        if not cam:
            return
        cam.setdefault("detected_objects", {})
        cam.setdefault("alerts", {})
        for cls_name in self.vehicle_classes:
            cam["detected_objects"][cls_name] = self.vehicle_actions[cls_name].isChecked()
            cam["alerts"][cls_name] = self.alert_actions[cls_name].isChecked()
        self.save_config()
        self.video_player.update_detections_config(self.get_checked_vehicle_actions())
        self.video_player.update_alerts_config(self.get_checked_alert_actions())
        if camera_id in self.camera_threads:
            worker = self.camera_threads[camera_id]
            worker.enabled_alerts = self.get_checked_alert_actions()

    def save_config(self):
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

    def _populate_main_camera_combo(self):
        self.camera_combo.clear()
        saved_cameras = self.config.get("cameras", [])
        if saved_cameras:
            for cam in saved_cameras:
                self.camera_combo.addItem(cam["name"], cam["id"])
            self.camera_combo.setEnabled(True)
        else:
            self.camera_combo.addItem("No cameras found", -1)
            self.camera_combo.setEnabled(False)

    def _apply_custom_theme(self):
        palette = QPalette()
        bg_color = QColor("#2F4F4F")
        base_color = QColor("#2B2B2B")
        text_color = QColor("#edf0f1")
        accent_color = QColor("#80cbc4")
        border_color = QColor("#5F9EA0")
        palette.setColor(QPalette.Window, bg_color)
        palette.setColor(QPalette.WindowText, text_color)
        palette.setColor(QPalette.Base, base_color)
        palette.setColor(QPalette.AlternateBase, bg_color)
        palette.setColor(QPalette.ToolTipBase, text_color)
        palette.setColor(QPalette.ToolTipText, text_color)
        palette.setColor(QPalette.Text, text_color)
        palette.setColor(QPalette.Button, QColor("#3f6d6b"))
        palette.setColor(QPalette.ButtonText, text_color)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Highlight, accent_color)
        palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(palette)
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {bg_color.name()};
            }}
            QLabel {{
                color: {text_color.name()};
            }}
            QToolButton {{
                background-color: #3f6d6b;
                color: {text_color.name()};
                font-size: 13px;
                border-radius: 4px;
                padding: 6px;
            }}
            QToolButton::menu-indicator {{
                image: none;
            }}
            QGroupBox {{
                border: 1px solid {border_color.name()};
                margin-top: 6px;
                border-radius: 4px;
            }}
            QGroupBox:title {{
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
            }}
            QPushButton {{
                background-color: #3f6d6b;
                color: {text_color.name()};
                font-size: 13px;
                border-radius: 4px;
                padding: 6px;
            }}
            QPushButton:hover {{
                background-color: #4b8280;
            }}
            QComboBox {{
                background-color: #3d5050;
                color: {text_color.name()};
                font-size: 13px;
                border-radius: 4px;
                padding: 4px;
            }}
            QSlider::groove:horizontal {{
                border: 1px solid #777;
                height: 6px;
                background: #3d5050;
                margin: 0px;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {accent_color.name()};
                border: 1px solid #5f9e9e;
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }}
            QMessageBox {{
                background-color: #f2f2f2;
            }}
            QMessageBox QLabel {{
                color: #000000;
            }}
            QMessageBox QPushButton {{
                background-color: #d5d5d5;
                color: #000000;
                border-radius: 3px;
                padding: 6px;
            }}
            QMessageBox QPushButton:hover {{
                background-color: #c5c5c5;
            }}
        """)


class CameraAddDialog(QDialog):
    def __init__(self, available_cameras, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add new camera")
        self.resize(400, 200)
        self.combo = QComboBox()
        for name, idx in available_cameras:
            self.combo.addItem(name, idx)
        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Select from available cameras:", self))
        layout.addWidget(self.combo)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.setStyleSheet("""
            QDialog {
                background-color: #f2f2f2;
                color: #000000;
            }
            QLabel {
                color: #000000;
            }
            QComboBox {
                background-color: #ffffff;
                color: #000000;
            }
            QPushButton {
                background-color: #dbdbdb;
                color: #000000;
                border-radius: 4px;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #cccccc;
            }
        """)

    def get_selected_camera(self):
        idx = self.combo.currentIndex()
        return (self.combo.itemText(idx), self.combo.itemData(idx))


class PhoneManagerDialog(QDialog):
    def __init__(self, phones_list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Phone management")
        self.resize(400, 300)
        self.phones = phones_list.copy()
        self.list_widget = QListWidget()
        for p in self.phones:
            self.list_widget.addItem(p)
        self.btn_add = QPushButton("Add")
        self.btn_add.clicked.connect(self.add_phone)
        self.btn_remove = QPushButton("Remove")
        self.btn_remove.clicked.connect(self.remove_phone)
        self.btn_ok = QPushButton("OK")
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        layout = QVBoxLayout()
        layout.addWidget(self.list_widget)
        hl = QHBoxLayout()
        hl.addWidget(self.btn_add)
        hl.addWidget(self.btn_remove)
        layout.addLayout(hl)
        hl2 = QHBoxLayout()
        hl2.addStretch(1)
        hl2.addWidget(self.btn_ok)
        hl2.addWidget(self.btn_cancel)
        layout.addLayout(hl2)
        self.setLayout(layout)
        self.setStyleSheet("""
            QDialog {
                background-color: #f2f2f2;
                color: #000000;
            }
            QLabel {
                color: #000000;
            }
            QListWidget {
                background-color: #ffffff;
                color: #000000;
            }
            QPushButton {
                background-color: #dbdbdb;
                color: #000000;
                border-radius: 4px;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #cccccc;
            }
        """)

    def is_valid_phone(self, phone_str: str) -> bool:
        pattern = re.compile(r'^\d{12}$')
        return bool(pattern.match(phone_str))

    def add_phone(self):
        text, ok = QInputDialog.getText(self, "New phone number", "Format 380XXXXXXXXX (12 digits):")
        if ok and text.strip():
            phone = text.strip()
            if not self.is_valid_phone(phone):
                QMessageBox.warning(self, "Error", "Invalid phone format (12 digits required).")
                return
            if phone in self.phones:
                QMessageBox.warning(self, "Warning", "This number already exists.")
            else:
                self.phones.append(phone)
                self.list_widget.addItem(phone)

    def remove_phone(self):
        items = self.list_widget.selectedItems()
        for it in items:
            ph = it.text()
            if ph in self.phones:
                self.phones.remove(ph)
            self.list_widget.takeItem(self.list_widget.row(it))

    def get_phones(self):
        return self.phones


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()