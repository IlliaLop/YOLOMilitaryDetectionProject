import cv2
from PyQt5.QtCore import QThread, pyqtSignal
import time
import os
import datetime

from ultralytics import YOLO
from twilio_messages import send_warning
from config import CFG
from s3 import upload_and_get_temporary_url

class CameraThread(QThread):
    camera_event = pyqtSignal(int, object)

    def __init__(self, cam_id, enabled_alerts=None, phones=None, parent=None):
        super().__init__(parent)
        self.cam_id = cam_id
        self.stop_flag = False
        self.interval = 1.0
        self.enabled_alerts = enabled_alerts or []
        self.phones = phones or []
        self.last_save_time = {}

    def run(self):
        cap = cv2.VideoCapture(self.cam_id)
        if not cap.isOpened():
            print(f"Could not open camera {self.cam_id}")
            return
        model = YOLO(CFG.WEIGHTS)
        save_dir = "saved_frames"
        os.makedirs(save_dir, exist_ok=True)

        while not self.stop_flag:
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"[CameraThread] Camera {self.cam_id} returned an empty frame.")
                time.sleep(self.interval)
                continue
            results = list(model(frame, CFG.CONFIDENCE))
            if len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls_id]
                    if label in self.enabled_alerts:
                        self.check_and_save(frame, box, label, conf, save_dir)
            self.camera_event.emit(self.cam_id, frame)
            time.sleep(self.interval)

        cap.release()
        print(f"[CameraThread] Camera {self.cam_id} finished.")

    def check_and_save(self, frame, box, label, conf, save_dir):
        cooldown = 600
        now = datetime.datetime.now()
        last_time = self.last_save_time.get(label)
        if last_time is None or (now - last_time).total_seconds() >= cooldown:
            self.last_save_time[label] = now
            print(f"[CameraThread] Saving frame for \"{label}\" (>=10 minutes passed).")
            x1, y1, x2, y2 = box.xyxy[0]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(
                frame, f"{label} {conf:.2f}",
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
            )
            filename = f"{label}_{now.strftime('%Y-%m-%d-%H-%M-%S')}.jpg"
            file_path = os.path.join(save_dir, filename)
            cv2.imwrite(file_path, frame)
            print(f"[CameraThread] Image saved: {file_path}")
            bucket_name = "diplomamodelstorage"
            url, s3_key = upload_and_get_temporary_url(file_path, bucket_name, 86400)
            print("Temporary link:", url)
            for phone in self.phones:
                send_warning(url, phone, label)

    def stop(self):
        self.stop_flag = True