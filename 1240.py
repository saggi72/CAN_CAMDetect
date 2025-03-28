# -*- coding: utf-8 -*-
import sys
import os
import datetime
import re
import time

# --- PyQt5 Imports ---
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QComboBox, QLabel, QLineEdit, QFileDialog,
                             QTextEdit, QStatusBar, QMessageBox, QCheckBox, QGroupBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QMutex, QMutexLocker, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QIntValidator, QTextCursor

# --- Other Imports ---
import cv2
try:
    import can
except ImportError:
    print("Lỗi: Thư viện 'python-can' chưa được cài đặt.")
    print("Vui lòng chạy: pip install python-can")
    sys.exit(1)
import numpy as np

# ---- Global Settings ----
DEFAULT_SAVE_DIR = os.path.join(os.path.expanduser("~"), "CameraCAN_Recordings")
CAMERA_SCAN_LIMIT = 5 # Quét từ index 0 đến 4
DEFAULT_BITRATE = 500000

os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)

# ---- Thread cho Camera ----
class CameraThread(QThread):
    changePixmap = pyqtSignal(QPixmap)
    recordingStartedSignal = pyqtSignal()
    recordingStoppedSignal = pyqtSignal(str)
    cameraErrorSignal = pyqtSignal(str)

    def __init__(self, camera_index, save_dir, parent=None): # Thay camera_source thành camera_index
        super().__init__(parent)
        self.camera_index = camera_index # Lưu index
        self.save_dir = save_dir
        self._running = False
        self._recording = False
        self.video_writer = None
        self.cap = None
        self.error_string_from_can = "UnknownEvent"
        self.temp_filename = None
        self.mutex = QMutex()

    def set_save_dir(self, directory):
        if directory and os.path.isdir(directory):
            with QMutexLocker(self.mutex):
                self.save_dir = directory
        else:
            print(f"CameraThread: Cảnh báo: Thư mục lưu '{directory}' không hợp lệ.")

    def run(self):
        self._running = True
        print(f"CameraThread: Attempting to open camera index: {self.camera_index}")
        try:
            # --- Đơn giản hóa việc mở camera ---
            # Thử mở trực tiếp bằng index, cách này thường hoạt động với webcam
            self.cap = cv2.VideoCapture(self.camera_index)

            if not self.cap or not self.cap.isOpened():
                # Thử thêm backend DSHOW cho Windows nếu lần đầu thất bại
                if os.name == 'nt':
                     print(f"CameraThread: Failed with default backend, trying CAP_DSHOW for index {self.camera_index}...")
                     self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

                # Nếu vẫn không mở được
                if not self.cap or not self.cap.isOpened():
                    raise ConnectionError(f"Không thể mở camera index {self.camera_index}")

            print(f"CameraThread: Camera index {self.camera_index} opened successfully.")

            # --- Lấy thông số Camera ---
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # Thử lấy FPS, nếu không được hoặc quá cao/thấp thì dùng mặc định
            try:
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                if not (0 < fps <= 120): # Check FPS trong khoảng hợp lý
                     print(f"CameraThread: Invalid FPS ({fps}) from camera index {self.camera_index}, using default 25.")
                     fps = 25
            except:
                 print(f"CameraThread: Could not get FPS for camera index {self.camera_index}, using default 25.")
                 fps = 25

            if frame_width <= 0 or frame_height <= 0:
                 # Thử lấy lại kích thước nếu lần đầu lỗi (một số camera cần thời gian)
                 time.sleep(0.5)
                 frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                 frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                 if frame_width <= 0 or frame_height <= 0:
                      raise ValueError(f"Không lấy được kích thước frame hợp lệ từ camera index {self.camera_index}.")

            print(f"CameraThread: Resolution: {frame_width}x{frame_height}, FPS: {fps:.2f}")

            # --- Vòng lặp đọc frame ---
            while self._running:
                ret, frame = self.cap.read()
                if not ret:
                    print(f"CameraThread: Warning: Failed to grab frame from index {self.camera_index}.")
                    if not self.cap.isOpened():
                         print(f"CameraThread: Error: Connection lost for camera index {self.camera_index}.")
                         self._running = False
                         self.cameraErrorSignal.emit(f"Mất kết nối camera index {self.camera_index}.")
                         break
                    time.sleep(0.05) # Đợi nếu đọc lỗi frame
                    continue

                # Xử lý hiển thị
                try:
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    convert_to_qt_format = QImage(rgb_image.data, w, h, w * ch, QImage.Format_RGB888)
                    if not convert_to_qt_format.isNull():
                        p = QPixmap.fromImage(convert_to_qt_format)
                        if not p.isNull(): self.changePixmap.emit(p)
                except Exception as display_e: print(f"CameraThread: Display error: {display_e}")

                # Xử lý ghi video
                with QMutexLocker(self.mutex):
                    if self._recording and self.video_writer and self.video_writer.isOpened():
                        try: self.video_writer.write(frame)
                        except Exception as write_e:
                            print(f"CameraThread: Write frame error: {write_e}")
                            self._recording = False
                            try: self.video_writer.release()
                            except Exception: pass
                            self.video_writer = None
                            self.cameraErrorSignal.emit(f"Lỗi ghi frame video: {write_e}")

                # Sleep để kiểm soát tốc độ
                sleep_duration = max(0.01, (1.0 / fps) * 0.9 if fps > 0 else 0.04) # 0.04s tương đương 25fps
                time.sleep(sleep_duration)

        except (ConnectionError, ValueError) as e:
             print(f"CameraThread: Initialization/runtime error: {e}")
             self.cameraErrorSignal.emit(str(e))
        except Exception as e:
             print(f"CameraThread: Unexpected error: {type(e).__name__}: {e}")
             self.cameraErrorSignal.emit(f"Lỗi camera không xác định: {e}")
        finally:
            # --- Dọn dẹp ---
            print(f"CameraThread: Finishing for index {self.camera_index}...")
            if self.cap:
                self.cap.release()
                print("CameraThread: cv2.VideoCapture released.")
            with QMutexLocker(self.mutex):
                if self.video_writer:
                     try:
                         if self._recording:
                            print("CameraThread: Releasing VideoWriter...")
                            self.video_writer.release()
                            print("CameraThread: VideoWriter released.")
                     except Exception as release_e:
                         print(f"CameraThread: Error releasing VideoWriter: {release_e}")
                     finally: self.video_writer = None
            self._recording = False
            print(f"CameraThread: Thread finished for index {self.camera_index}.")

    def stop(self):
        print(f"CameraThread: Stop request for index {self.camera_index}.")
        self._running = False
        if self.isRunning():
             if not self.wait(3000): # Đợi tối đa 3 giây
                 print(f"CameraThread: Warning: Thread for index {self.camera_index} unresponsive. Terminating.")
                 self.terminate()

    # --- start_recording và stop_recording_and_save giữ nguyên logic như trước ---
    # (Chúng không cần thay đổi khi bỏ cam IP)
    def start_recording(self):
        with QMutexLocker(self.mutex):
            if self._recording:
                print("CameraThread: Start recording called, but already recording.")
                return False
            if not self.cap or not self.cap.isOpened():
                err_msg = "Lỗi: Không thể bắt đầu ghi hình vì camera chưa sẵn sàng."
                print(f"CameraThread: {err_msg}")
                return False

            if not self.save_dir or not os.path.isdir(self.save_dir):
                err_msg = f"Lỗi: Thư mục lưu '{self.save_dir}' không hợp lệ."
                print(f"CameraThread: {err_msg}")
                self.cameraErrorSignal.emit(err_msg)
                return False

            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if not (0 < fps <= 120): fps = 25
            if frame_width <= 0 or frame_height <= 0:
                err_msg = "Lỗi: Không lấy được kích thước frame hợp lệ."
                print(f"CameraThread: {err_msg}")
                self.cameraErrorSignal.emit(err_msg)
                return False

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            self.temp_filename = os.path.join(self.save_dir, f"rec_{timestamp}_temp.mp4")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            try:
                print(f"CameraThread: Starting recording -> {self.temp_filename} ({frame_width}x{frame_height} @ {fps:.2f}fps)")
                self.video_writer = cv2.VideoWriter(self.temp_filename, fourcc, fps, (frame_width, frame_height))

                if not self.video_writer.isOpened():
                     print("CameraThread: Warning: Failed with mp4v, trying XVID...")
                     fourcc = cv2.VideoWriter_fourcc(*'XVID')
                     self.video_writer = cv2.VideoWriter(self.temp_filename, fourcc, fps, (frame_width, frame_height))
                     if not self.video_writer.isOpened():
                          raise IOError("Không thể mở VideoWriter (mp4v, XVID).")

                self._recording = True
                self.recordingStartedSignal.emit()
                print("CameraThread: Recording started signal.")
                return True

            except Exception as e:
                err_msg = f"Lỗi VideoWriter: {e}"
                print(f"CameraThread: {err_msg}")
                self.cameraErrorSignal.emit(err_msg)
                if self.video_writer:
                    try: self.video_writer.release()
                    except: pass
                self.video_writer = None
                self._recording = False
                if self.temp_filename and os.path.exists(self.temp_filename):
                     try: os.remove(self.temp_filename); print(f"Removed failed temp: {self.temp_filename}")
                     except: pass
                self.temp_filename = None
                return False

    def stop_recording_and_save(self, error_string="UnknownEvent"):
        final_filepath = ""
        writer_instance = None # Biến tạm để lưu writer
        temp_file_to_rename = None # Biến tạm để lưu tên file

        with QMutexLocker(self.mutex):
            if not self._recording:
                print("CameraThread: Stop called but not recording.")
                if self.video_writer: # Dọn dẹp nếu còn sót
                    try: self.video_writer.release()
                    except: pass
                    self.video_writer = None
                if self.temp_filename and os.path.exists(self.temp_filename):
                    try: os.remove(self.temp_filename)
                    except: pass
                    self.temp_filename = None
                self.recordingStoppedSignal.emit("") # Vẫn báo dừng
                return

            print(f"CameraThread: Stopping recording. Event: '{error_string}'")
            self._recording = False
            writer_instance = self.video_writer
            temp_file_to_rename = self.temp_filename
            self.video_writer = None
            self.temp_filename = None
        # ---- Hết vùng khóa Mutex ----

        try:
            if writer_instance:
                print("CameraThread: Releasing VideoWriter...")
                writer_instance.release()
                print("CameraThread: VideoWriter released.")
            else:
                temp_file_to_rename = None # Không có writer thì không có file tạm

            if temp_file_to_rename and os.path.exists(temp_file_to_rename):
                file_size = os.path.getsize(temp_file_to_rename)
                print(f"CameraThread: Temp file size: {file_size} bytes")
                if file_size > 500: # Lưu nếu kích thước lớn hơn 500 bytes (ngưỡng nhỏ)
                    safe_error_string = re.sub(r'[\\/*?:"<>|]', "_", error_string)
                    safe_error_string = "_".join(safe_error_string.split()).strip('_')
                    if not safe_error_string: safe_error_string = "Event"

                    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
                    base_fn = f"{date_str}_{safe_error_string}.mp4"
                    final_fn = base_fn
                    # Đảm bảo self.save_dir là hợp lệ
                    current_save_dir = self.save_dir if os.path.isdir(self.save_dir) else DEFAULT_SAVE_DIR
                    final_filepath = os.path.join(current_save_dir, final_fn)

                    counter = 1
                    while os.path.exists(final_filepath):
                          name, ext = os.path.splitext(base_fn)
                          final_fn = f"{name}_{counter}{ext}"
                          final_filepath = os.path.join(current_save_dir, final_fn)
                          counter += 1
                          if counter > 100: final_filepath = ""; break # Tránh lặp vô hạn

                    if final_filepath:
                        print(f"CameraThread: Renaming '{temp_file_to_rename}' to '{final_filepath}'")
                        os.rename(temp_file_to_rename, final_filepath)
                        print(f"CameraThread: Saved: {final_filepath}")
                    else:
                        print(f"CameraThread: Could not generate unique filename. Deleting temp: {temp_file_to_rename}")
                        os.remove(temp_file_to_rename)
                else:
                    print(f"CameraThread: Temp file '{temp_file_to_rename}' too small or empty. Deleting.")
                    os.remove(temp_file_to_rename)
                    final_filepath = ""
            else:
                print("CameraThread: No valid temp file found to save.")
                final_filepath = ""

        except Exception as e:
            print(f"CameraThread: Error saving/renaming: {e}")
            final_filepath = ""
            if temp_file_to_rename and os.path.exists(temp_file_to_rename):
                 try: os.remove(temp_file_to_rename); print("Cleaned up temp file on error.")
                 except: pass
        finally:
             self.recordingStoppedSignal.emit(final_filepath)
             print(f"CameraThread: Stopped signal emitted. Path='{final_filepath}'")


# ---- Thread cho CAN ----
# (Giữ nguyên CanThread như phiên bản trước - không cần thay đổi)
class CanThread(QThread):
    logMessage = pyqtSignal(str)
    startRecordingSignal = pyqtSignal()
    stopRecordingAndSaveSignal = pyqtSignal(str) # Gửi chuỗi lỗi/sự kiện từ payload
    canErrorSignal = pyqtSignal(str)
    connectionStatusSignal = pyqtSignal(bool) # True khi kết nối, False khi ngắt

    def __init__(self, interface, channel, bitrate, start_id_hex, stop_id_hex, emergency_id_hex=None, parent=None):
        super().__init__(parent)
        self.interface = interface
        self.channel = channel
        self.bitrate = bitrate
        self.start_id = None
        self.stop_id = None
        self.emergency_id = None
        self._running = False
        self.bus = None
        self.notifier = None
        self.listener = None

        print(f"CanThread: Initializing with config: IF='{interface}', CH='{channel}', BR={bitrate}, Start='{start_id_hex}', Stop='{stop_id_hex}'")

        missing_ids = []
        try:
            if not start_id_hex: missing_ids.append("Bắt Đầu Ghi")
            else: self.start_id = int(start_id_hex, 16)

            if not stop_id_hex: missing_ids.append("Dừng & Lưu")
            else: self.stop_id = int(stop_id_hex, 16)

            if emergency_id_hex: self.emergency_id = int(emergency_id_hex, 16)

            if missing_ids: raise ValueError(f"CAN ID bắt buộc bị thiếu: {', '.join(missing_ids)}")
            if not isinstance(self.bitrate, int) or self.bitrate <= 0: raise ValueError("Bitrate không hợp lệ.")

            print(f"CanThread: CAN IDs parsed - Start: {hex(self.start_id)}, Stop: {hex(self.stop_id)}, Emergency: {hex(self.emergency_id) if self.emergency_id else 'None'}")

        except ValueError as e:
            raise ValueError(f"Lỗi cấu hình CAN: {e}")


    def run(self):
        self._running = True
        try:
            print("CanThread: Attempting to connect to CAN bus...")
            kwargs = {'interface': self.interface, 'channel': self.channel, 'receive_own_messages': False}
            # Các interface thường dùng với CANable/SLCAN cần bitrate
            if self.interface.lower() in ['slcan', 'serial', 'pcan', 'kvaser', 'vector', 'ixxat', 'usb2can']:
                 kwargs['bitrate'] = self.bitrate
            elif self.interface.lower() == 'socketcan': # Trên Linux
                 print(f"CanThread: Note for socketcan - Bitrate ({self.bitrate}) should be set externally (e.g., using 'ip link').")

            print(f"CanThread: Initializing can.interface.Bus with: {kwargs}")
            self.bus = can.interface.Bus(**kwargs)
            print(f"CanThread: CAN bus connected! Type: {self.bus.__class__.__name__}")
            self.connectionStatusSignal.emit(True)

            class MyListener(can.Listener):
                def __init__(self, parent_thread):
                    self.parent = parent_thread

                def on_message_received(self, msg: can.Message):
                    if not self.parent._running: return

                    log_str = f"ID: {msg.arbitration_id:<5X} DLC: {msg.dlc} Data: {msg.data.hex().upper():<18}"
                    self.parent.logMessage.emit(log_str)

                    try:
                        if self.parent.start_id is not None and msg.arbitration_id == self.parent.start_id:
                            print(f"CanThread: Rx Start Rec. (ID: {msg.arbitration_id:#X})")
                            self.parent.startRecordingSignal.emit()
                        elif self.parent.stop_id is not None and msg.arbitration_id == self.parent.stop_id:
                            print(f"CanThread: Rx Stop Rec. (ID: {msg.arbitration_id:#X})")
                            payload_str = "PayloadError"
                            try:
                                payload_bytes = msg.data
                                null_index = payload_bytes.find(b'\x00')
                                if null_index != -1: payload_bytes = payload_bytes[:null_index]
                                payload_str = payload_bytes.decode('utf-8', errors='replace').strip()
                                if not payload_str: payload_str = "EventDataEmpty"
                            except Exception as decode_err:
                                print(f"CanThread: Payload decode error: {decode_err}")
                            print(f"CanThread: Extracted event: '{payload_str}'")
                            self.parent.stopRecordingAndSaveSignal.emit(payload_str)
                    except Exception as handler_err:
                         print(f"CanThread: Error handling msg: {handler_err}")

                def on_error(self, exc):
                     print(f"CanThread Listener Error: {exc}")
                     if self.parent._running:
                        self.parent.canErrorSignal.emit(f"Lỗi Bus/Listener: {exc}")

            self.listener = MyListener(self)
            self.notifier = can.Notifier(self.bus, [self.listener], timeout=1.0)
            print("CanThread: CAN Notifier started.")

            while self._running:
                time.sleep(0.2) # Giữ thread chạy, giảm CPU

        except can.CanError as e:
            err_msg = f"Lỗi kết nối/giao tiếp CAN: {e}"
            print(f"CanThread: {err_msg}")
            if self._running: self.canErrorSignal.emit(err_msg)
        except ValueError as e: # Lỗi cấu hình (bitrate/ID)
            err_msg = f"Lỗi cấu hình CAN: {e}"
            print(f"CanThread: {err_msg}")
            if self._running: self.canErrorSignal.emit(err_msg)
        except Exception as e:
            err_msg = f"Lỗi CAN không xác định: {type(e).__name__}: {e}"
            print(f"CanThread: {err_msg}")
            if self._running: self.canErrorSignal.emit(err_msg)
        finally:
            print("CanThread: Finishing...")
            self._running = False
            if self.notifier:
                try: self.notifier.stop(timeout=1.0) # Timeout ngắn hơn
                except Exception as e: print(f"CanThread: Error stop notifier: {e}")
            if self.bus:
                try: self.bus.shutdown()
                except Exception as e: print(f"CanThread: Error shutdown bus: {e}")
            self.connectionStatusSignal.emit(False)
            print("CanThread: Thread finished.")

    def stop(self):
        print("CanThread: Stop request received.")
        self._running = False


# ---- Giao diện chính (Đã đơn giản hóa) ----
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Giảm tiêu đề và kích thước mặc định
        self.setWindowTitle("Camera CAN Simple Controller")
        self.setGeometry(150, 150, 950, 700)

        # Thuộc tính
        self.camera_thread = None
        self.can_thread = None
        self.current_save_dir = DEFAULT_SAVE_DIR
        self.is_recording_flag = False
        self.can_log_enabled = True

        # Layout chính
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # == Cột Trái: Video ==
        video_layout = QVBoxLayout()
        video_groupbox = QGroupBox("Camera View")
        video_gb_layout = QVBoxLayout()
        self.video_label = QLabel("Camera Tắt")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(600, 450) # Giảm kích thước min
        self.video_label.setStyleSheet("border: 1px solid gray; background-color: #ddd; color: black;")
        video_gb_layout.addWidget(self.video_label)
        video_groupbox.setLayout(video_gb_layout)
        video_layout.addWidget(video_groupbox)
        main_layout.addLayout(video_layout, 3) # Video chiếm nhiều không gian hơn

        # == Cột Phải: Controls ==
        control_layout = QVBoxLayout()

        # -- 1. Camera Control --
        cam_gb = QGroupBox("1. Điều Khiển Camera")
        cam_v_layout = QVBoxLayout()
        cam_sel_layout = QHBoxLayout()
        self.cam_combo = QComboBox()
        self.cam_combo.setToolTip("Chọn webcam có sẵn")
        self.scan_cam_btn = QPushButton("Quét Lại")
        self.scan_cam_btn.setToolTip("Tìm lại webcam trên máy")
        self.scan_cam_btn.clicked.connect(self.scan_cameras)
        cam_sel_layout.addWidget(QLabel("Webcam:"))
        cam_sel_layout.addWidget(self.cam_combo, 1)
        cam_sel_layout.addWidget(self.scan_cam_btn)
        cam_v_layout.addLayout(cam_sel_layout)
        # --- Bỏ phần thêm IP Cam ---
        cam_btn_layout = QHBoxLayout()
        self.start_cam_btn = QPushButton("Bật Camera")
        self.start_cam_btn.setStyleSheet("background-color: lightgreen; font-weight: bold;")
        self.start_cam_btn.clicked.connect(self.start_camera)
        self.stop_cam_btn = QPushButton("Tắt Camera")
        self.stop_cam_btn.setStyleSheet("background-color: salmon; font-weight: bold;")
        self.stop_cam_btn.clicked.connect(self.stop_camera)
        self.stop_cam_btn.setEnabled(False)
        cam_btn_layout.addWidget(self.start_cam_btn)
        cam_btn_layout.addWidget(self.stop_cam_btn)
        cam_v_layout.addLayout(cam_btn_layout)
        cam_gb.setLayout(cam_v_layout)
        control_layout.addWidget(cam_gb)

        # -- 2. System Control --
        sys_gb = QGroupBox("2. Lưu Trữ & Log")
        sys_v_layout = QVBoxLayout()
        # Thư mục lưu
        dir_layout = QHBoxLayout()
        self.dir_label = QLineEdit(self.current_save_dir)
        self.dir_label.setReadOnly(True)
        self.select_dir_btn = QPushButton("Thư Mục Lưu") # Rút gọn tên nút
        self.select_dir_btn.setToolTip("Chọn nơi lưu video khi ghi bằng lệnh CAN")
        self.select_dir_btn.clicked.connect(self.select_directory)
        dir_layout.addWidget(QLabel("Lưu vào:"))
        dir_layout.addWidget(self.dir_label, 1)
        dir_layout.addWidget(self.select_dir_btn)
        sys_v_layout.addLayout(dir_layout)
        # Log CAN
        self.can_log_display = QTextEdit() # Tạo log display trước
        self.can_log_display.setReadOnly(True)
        self.can_log_display.setFontFamily("Consolas")
        self.can_log_display.setLineWrapMode(QTextEdit.NoWrap) # Không xuống dòng tự động
        self.can_log_display.setFixedHeight(100) # Giới hạn chiều cao log
        log_ctrl_layout = QHBoxLayout()
        self.can_log_checkbox = QCheckBox("Hiện Log CAN")
        self.can_log_checkbox.setChecked(self.can_log_enabled)
        self.can_log_checkbox.toggled.connect(self.toggle_can_logging) # Kết nối signal toggled
        clear_log_btn = QPushButton("Xóa Log")
        clear_log_btn.clicked.connect(self.can_log_display.clear)
        log_ctrl_layout.addWidget(self.can_log_checkbox)
        log_ctrl_layout.addStretch()
        log_ctrl_layout.addWidget(clear_log_btn)
        sys_v_layout.addLayout(log_ctrl_layout)
        sys_v_layout.addWidget(self.can_log_display) # Thêm log display vào layout
        sys_gb.setLayout(sys_v_layout)
        control_layout.addWidget(sys_gb)

        # -- 3. CAN Control --
        can_gb = QGroupBox("3. Điều Khiển Qua CAN")
        can_v_layout = QVBoxLayout()
        can_hw_layout = QHBoxLayout()
        self.can_interface_input = QLineEdit("slcan")
        self.can_interface_input.setToolTip("Interface: slcan (cho CANable), socketcan (Linux),...")
        self.can_channel_input = QLineEdit("COM3" if os.name == 'nt' else "/dev/ttyACM0") # Đổi mặc định cho Windows
        self.can_channel_input.setToolTip("Channel: Cổng COM (Win) hoặc tty (Linux) của CANable")
        self.can_bitrate_input = QLineEdit(str(DEFAULT_BITRATE))
        self.can_bitrate_input.setToolTip("Bitrate mạng CAN")
        self.can_bitrate_input.setValidator(QIntValidator(1, 1000000))
        can_hw_layout.addWidget(QLabel("IF:"))
        can_hw_layout.addWidget(self.can_interface_input)
        can_hw_layout.addWidget(QLabel("Ch:"))
        can_hw_layout.addWidget(self.can_channel_input)
        can_hw_layout.addWidget(QLabel("BR:"))
        can_hw_layout.addWidget(self.can_bitrate_input)
        can_v_layout.addLayout(can_hw_layout)
        can_id_layout = QHBoxLayout()
        self.start_id_input = QLineEdit("100")
        self.start_id_input.setPlaceholderText("Start Rec ID (Hex)")
        self.stop_id_input = QLineEdit("101")
        self.stop_id_input.setPlaceholderText("Stop Rec ID (Hex)")
        can_id_layout.addWidget(QLabel("Start:"))
        can_id_layout.addWidget(self.start_id_input)
        can_id_layout.addWidget(QLabel("Stop:"))
        can_id_layout.addWidget(self.stop_id_input)
        can_v_layout.addLayout(can_id_layout)
        self.connect_can_btn = QPushButton("Kết Nối CAN")
        self.connect_can_btn.setCheckable(True)
        self.connect_can_btn.setStyleSheet("font-weight: bold;")
        self.connect_can_btn.clicked.connect(self.toggle_can_connection)
        can_v_layout.addWidget(self.connect_can_btn)
        can_gb.setLayout(can_v_layout)
        control_layout.addWidget(can_gb)

        control_layout.addStretch() # Đẩy các group box lên trên
        main_layout.addLayout(control_layout, 1) # Control chiếm ít không gian hơn

        # Status Bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        # Khởi tạo
        self.scan_cameras() # Quét camera khi bắt đầu
        # Cập nhật status ban đầu dựa trên kết quả quét
        if self.cam_combo.count() > 0 and self.cam_combo.itemText(0) != "Không tìm thấy camera":
            self.statusBar.showMessage("Sẵn sàng. Chọn Webcam và nhấn Bật Camera.")
        else:
            self.statusBar.showMessage("Không tìm thấy webcam. Kiểm tra kết nối và nhấn Quét Lại.")


    # --- Các Phương Thức ---

    def scan_cameras(self):
        """Chỉ quét webcam."""
        self.statusBar.showMessage("Đang quét webcam...")
        self.setEnabled_CameraControls(False) # Khóa control khi quét
        QApplication.processEvents()
        time.sleep(0.1) # Chờ chút trước khi quét

        # --- Chỉ quét hardware camera ---
        self.cam_combo.clear() # Xóa hết item cũ
        found_cameras = []
        print("Scanning for webcams...")
        for i in range(CAMERA_SCAN_LIMIT):
            print(f"  Trying index {i}...")
            cap_test = None
            try:
                cap_test = cv2.VideoCapture(i) # Thử cách đơn giản nhất
                if cap_test and cap_test.isOpened():
                     cam_name = f"Webcam {i}"
                     found_cameras.append((cam_name, i)) # Lưu cả tên và index (int)
                     print(f"    -> Found: {cam_name}")
                # Không cần thử backend khác nếu cách đơn giản hoạt động
            except Exception as e:
                 print(f"    -> Error opening index {i}: {e}")
            finally:
                if cap_test:
                    cap_test.release()
                    # print(f"    -> Released index {i}")
                # Thêm độ trễ nhỏ giữa các lần thử để tránh tranh chấp tài nguyên
                time.sleep(0.1) # Delay 100ms

        self.setEnabled_CameraControls(True) # Mở khóa control

        if not found_cameras:
            self.cam_combo.addItem("Không tìm thấy webcam")
            self.cam_combo.setEnabled(False)
            self.start_cam_btn.setEnabled(False)
            self.statusBar.showMessage("Không tìm thấy webcam. Kiểm tra và Quét Lại.")
            print("Webcam scan finished: None found.")
        else:
             self.cam_combo.setEnabled(True)
             for name, index in found_cameras: # userData là index (int)
                 self.cam_combo.addItem(name, userData=index)
             self.start_cam_btn.setEnabled(True)
             self.statusBar.showMessage("Sẵn sàng. Chọn Webcam và nhấn Bật Camera.")
             print(f"Webcam scan finished: Found {len(found_cameras)} webcam(s).")


    def setEnabled_CameraControls(self, enabled):
        """Bật/tắt control liên quan đến camera."""
        # Chỉ bật combo nếu có > 0 item và item đầu tiên không phải "Không tìm thấy"
        has_cam = self.cam_combo.count() > 0 and self.cam_combo.itemText(0) != "Không tìm thấy webcam"
        self.cam_combo.setEnabled(enabled and has_cam)
        self.scan_cam_btn.setEnabled(enabled)

        is_cam_running = self.camera_thread is not None and self.camera_thread.isRunning()
        # Nút Start chỉ bật khi enabled=True, có cam, và cam chưa chạy
        self.start_cam_btn.setEnabled(enabled and has_cam and not is_cam_running)
        # Nút Stop chỉ bật khi cam đang chạy
        self.stop_cam_btn.setEnabled(is_cam_running) # Sẽ bị disable khi start_camera gọi setEnable(False)

    # --- Bỏ hàm add_ip_camera ---

    def start_camera(self):
        """Kết nối và bắt đầu hiển thị webcam đã chọn."""
        if self.camera_thread and self.camera_thread.isRunning(): return

        selected_index = self.cam_combo.currentIndex()
        camera_index = self.cam_combo.itemData(selected_index) # Lấy index (int) từ userData
        current_cam_text = self.cam_combo.currentText()

        if selected_index < 0 or camera_index is None or current_cam_text == "Không tìm thấy webcam":
            QMessageBox.warning(self, "Lỗi", "Chưa chọn webcam hợp lệ!")
            return

        self.statusBar.showMessage(f"Đang kết nối {current_cam_text}...")
        self.setEnabled_CameraControls(False) # Khóa control
        self.stop_cam_btn.setEnabled(True)  # Bật nút stop ngay
        QApplication.processEvents()

        if self.camera_thread: # Dọn thread cũ nếu có
            self.camera_thread.stop() # Đảm bảo thread cũ dừng hẳn

        self.camera_thread = CameraThread(camera_index, self.current_save_dir) # Truyền index
        self.camera_thread.changePixmap.connect(self.set_image)
        self.camera_thread.recordingStartedSignal.connect(self.on_recording_started)
        self.camera_thread.recordingStoppedSignal.connect(self.on_recording_stopped)
        self.camera_thread.cameraErrorSignal.connect(self.on_camera_error)
        self.camera_thread.finished.connect(self.on_camera_thread_finished)

        print(f"MainWindow: Starting camera thread for index: {camera_index}")
        self.camera_thread.start()
        self.video_label.setText(f"Đang kết nối {current_cam_text}...")
        self.video_label.setStyleSheet("border: 2px solid orange;") # Viền cam đậm hơn


    def stop_camera(self):
        """Dừng webcam."""
        if self.camera_thread and self.camera_thread.isRunning():
            self.statusBar.showMessage("Đang dừng camera...")
            self.setEnabled_CameraControls(False) # Khóa nút khi đang dừng
            QApplication.processEvents()
            print("MainWindow: Requesting camera thread stop.")
            self.camera_thread.stop()
            # UI reset trong on_camera_thread_finished
        else:
            print("Stop camera called but thread not running.")
            self.on_camera_thread_finished() # Vẫn reset UI

    def on_camera_thread_finished(self):
        """Xử lý khi thread camera kết thúc."""
        print("MainWindow: Camera thread finished.")
        was_recording = self.is_recording_flag
        self.camera_thread = None
        self.is_recording_flag = False

        self.video_label.setText("Camera Tắt")
        self.video_label.setPixmap(QPixmap()) # Xóa ảnh
        self.video_label.setStyleSheet("border: 1px solid gray; background-color: #ddd; color: black;")
        self.setEnabled_CameraControls(True) # Mở khóa control

        current_status = self.statusBar.currentMessage()
        if "LỖI CAMERA" not in current_status:
            self.statusBar.showMessage("Camera đã tắt.")


    @pyqtSlot(QPixmap)
    def set_image(self, pixmap):
        """Hiển thị frame ảnh."""
        # Chỉ cập nhật nếu thread và label còn tồn tại
        if self.video_label and self.camera_thread and self.camera_thread.isRunning():
            current_style = self.video_label.styleSheet()
            # Đổi viền khi frame đầu tiên về (không ghi hình)
            if "orange" in current_style and not self.is_recording_flag:
                 self.video_label.setStyleSheet("border: 1px solid green;") # Viền xanh
                 if "Đang kết nối" in self.statusBar.currentMessage():
                     self.statusBar.showMessage(f"Đang hiển thị: {self.cam_combo.currentText()}")
            elif "green" in current_style and self.is_recording_flag:
                 # Trường hợp bắt đầu ghi khi đang hiển thị -> đổi sang đỏ
                 self.video_label.setStyleSheet("border: 3px solid red;")

            # Scale và hiển thị ảnh
            try:
                 scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                 self.video_label.setPixmap(scaled_pixmap)
            except Exception as e:
                 print(f"Error scaling/setting pixmap: {e}")


    def select_directory(self):
        """Chọn thư mục lưu."""
        new_dir = QFileDialog.getExistingDirectory(self, "Chọn Thư Mục Lưu Video", self.current_save_dir)
        if new_dir and os.path.isdir(new_dir) and new_dir != self.current_save_dir:
            self.current_save_dir = new_dir
            self.dir_label.setText(new_dir)
            if self.camera_thread and self.camera_thread.isRunning():
                self.camera_thread.set_save_dir(new_dir)
            print(f"Save directory set to: {new_dir}")
        elif new_dir:
             QMessageBox.warning(self, "Lỗi", f"Đường dẫn không hợp lệ: '{new_dir}'")


    # --- Các hàm xử lý CAN giữ nguyên như trước ---
    def toggle_can_connection(self):
        """Kết nối/Ngắt kết nối CAN."""
        if self.connect_can_btn.isChecked(): # Muốn kết nối
            if self.can_thread and self.can_thread.isRunning(): return
            # Validate Inputs
            interface = self.can_interface_input.text().strip()
            channel = self.can_channel_input.text().strip()
            bitrate_str = self.can_bitrate_input.text().strip()
            start_id_hex = self.start_id_input.text().strip()
            stop_id_hex = self.stop_id_input.text().strip()
            errors, bitrate = self.validate_can_inputs(interface, channel, bitrate_str, start_id_hex, stop_id_hex)
            if errors:
                 QMessageBox.warning(self, "Lỗi Cấu Hình CAN", f"Kiểm tra: {', '.join(errors)}")
                 self.connect_can_btn.setChecked(False)
                 return
            # Kết nối
            self.statusBar.showMessage(f"Đang kết nối CAN ({interface}/{channel})...")
            self.set_can_config_enabled(False) # Khóa input
            QApplication.processEvents()
            try:
                self.can_thread = CanThread(interface, channel, bitrate, start_id_hex, stop_id_hex)
                # Signals/Slots
                self.can_thread.logMessage.connect(self.append_can_log)
                self.can_thread.startRecordingSignal.connect(self.handle_start_recording_can)
                self.can_thread.stopRecordingAndSaveSignal.connect(self.handle_stop_recording_can)
                self.can_thread.canErrorSignal.connect(self.on_can_error)
                self.can_thread.connectionStatusSignal.connect(self.on_can_connection_status)
                self.can_thread.finished.connect(self.on_can_thread_finished)
                print("MainWindow: Starting CAN thread.")
                self.can_thread.start()
            except Exception as e:
                 self.on_can_error(f"Lỗi khởi tạo CAN: {e}")
        else: # Muốn ngắt kết nối
             if self.can_thread and self.can_thread.isRunning():
                 self.statusBar.showMessage("Đang ngắt kết nối CAN...")
                 self.connect_can_btn.setEnabled(False) # Khóa nút tạm thời
                 QApplication.processEvents()
                 self.can_thread.stop()
             else: self.on_can_thread_finished() # Reset UI

    def validate_can_inputs(self, interface, channel, bitrate_str, start_id_hex, stop_id_hex):
         """Helper kiểm tra input cấu hình CAN."""
         errors = []
         bitrate = 0
         if not interface: errors.append("Interface")
         if not channel: errors.append("Channel")
         if not bitrate_str: errors.append("Bitrate")
         else:
             try: bitrate = int(bitrate_str); assert bitrate > 0
             except: errors.append("Bitrate (Số > 0)")
         if not start_id_hex: errors.append("Start ID")
         else:
             try: int(start_id_hex, 16)
             except: errors.append("Start ID (Hex)")
         if not stop_id_hex: errors.append("Stop ID")
         else:
             try: int(stop_id_hex, 16)
             except: errors.append("Stop ID (Hex)")
         return errors, bitrate

    def set_can_config_enabled(self, enabled):
        """Bật/tắt input cấu hình CAN."""
        self.can_interface_input.setEnabled(enabled)
        self.can_channel_input.setEnabled(enabled)
        self.can_bitrate_input.setEnabled(enabled)
        self.start_id_input.setEnabled(enabled)
        self.stop_id_input.setEnabled(enabled)
        self.connect_can_btn.setEnabled(enabled) # Nút connect cũng bị khóa/mở

    def on_can_thread_finished(self):
        print("MainWindow: CAN thread finished.")
        self.can_thread = None
        if self.connect_can_btn.isChecked(): # Nếu dừng khi đang check -> lỗi
            self.connect_can_btn.setChecked(False)
        self.connect_can_btn.setText("Kết Nối CAN")
        self.set_can_config_enabled(True)
        if "LỖI CAN" not in self.statusBar.currentMessage():
             current_msg = self.statusBar.currentMessage()
             if "Đang ngắt" in current_msg or "Đã kết nối" in current_msg :
                 self.statusBar.showMessage("Đã ngắt kết nối CAN.")

    def on_can_connection_status(self, connected):
        print(f"MainWindow: CAN Connection Status -> {connected}")
        if connected:
            self.connect_can_btn.setText("Ngắt Kết Nối CAN")
            self.connect_can_btn.setChecked(True)
            self.statusBar.showMessage(f"Đã kết nối CAN ({self.can_interface_input.text()})", 0)
            self.set_can_config_enabled(False) # Khóa config
            self.connect_can_btn.setEnabled(True) # Vẫn cho ngắt
        else:
             # Gọi finished để reset UI
             # Không gọi trực tiếp vì finished signal đã được kết nối
             # print("Resetting CAN UI due to disconnection/failure.")
             # self.on_can_thread_finished() # Có thể gây gọi 2 lần, nên bỏ qua
             pass # on_can_thread_finished sẽ được gọi khi thread thực sự kết thúc

    def toggle_can_logging(self, checked):
        """Bật/tắt log dựa trên checkbox state."""
        self.can_log_enabled = checked
        print(f"CAN logging {'enabled' if checked else 'disabled'}")

    def append_can_log(self, message):
        if self.can_log_enabled:
            timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
            log_line = f"[{timestamp}] {message}"
            # Giữ giới hạn số dòng
            MAX_LOG_LINES = 500 # Giảm số dòng để nhẹ hơn
            doc = self.can_log_display.document()
            if doc.blockCount() > MAX_LOG_LINES:
                cursor = QTextCursor(doc)
                cursor.movePosition(QTextCursor.Start, QTextCursor.MoveAnchor)
                cursor.movePosition(QTextCursor.Down, QTextCursor.KeepAnchor, doc.blockCount() - MAX_LOG_LINES)
                cursor.removeSelectedText()
            # Thêm dòng mới và cuộn xuống
            self.can_log_display.append(log_line)


    def handle_start_recording_can(self):
        print("MainWindow: Rx Signal Start Recording")
        if self.camera_thread and self.camera_thread.isRunning() and not self.is_recording_flag:
            print(" -> Requesting camera start recording")
            self.camera_thread.start_recording()
        elif not self.camera_thread or not self.camera_thread.isRunning():
            self.statusBar.showMessage("CAN: Lệnh Ghi bị bỏ qua (Cam chưa bật)", 2500)
        # else: Đã đang ghi rồi, không cần làm gì

    def handle_stop_recording_can(self, event_string):
        print(f"MainWindow: Rx Signal Stop Recording (Event: '{event_string}')")
        if self.camera_thread and self.camera_thread.isRunning() and self.is_recording_flag:
            print(" -> Requesting camera stop recording")
            # Reset cờ ngay, gọi hàm stop trong thread
            self.is_recording_flag = False
            self.camera_thread.stop_recording_and_save(event_string)
        elif not self.camera_thread or not self.camera_thread.isRunning():
            self.statusBar.showMessage("CAN: Lệnh Dừng bị bỏ qua (Cam chưa bật)", 2500)
        # else: Không đang ghi, không cần làm gì

    def on_recording_started(self):
         print("MainWindow: Confirmed Recording Started")
         if self.camera_thread and self.camera_thread.isRunning():
             self.is_recording_flag = True
             self.statusBar.showMessage("ĐANG GHI HÌNH...", 0)
             self.video_label.setStyleSheet("border: 3px solid red;")
         else: print("Warning: Rec started signal but cam stopped.")

    def on_recording_stopped(self, saved_filepath):
         print(f"MainWindow: Confirmed Recording Stopped. Path: '{saved_filepath}'")
         # Cờ đã tắt ở handle_stop_recording_can
         if self.camera_thread and self.camera_thread.isRunning() and "LỖI" not in self.statusBar.currentMessage():
             self.video_label.setStyleSheet("border: 1px solid green;")
         if saved_filepath:
              self.statusBar.showMessage(f"Đã lưu: {os.path.basename(saved_filepath)}", 4000)
         elif "LỖI" not in self.statusBar.currentMessage():
              self.statusBar.showMessage("Đã dừng ghi (Không lưu file).", 4000)

    def on_camera_error(self, error_message):
        print(f"MainWindow: Rx Camera Error: {error_message}")
        QMessageBox.critical(self, "Lỗi Camera", error_message)
        self.statusBar.showMessage(f"LỖI CAMERA! {error_message}", 0)
        # on_camera_thread_finished sẽ reset UI

    def on_can_error(self, error_message):
        print(f"MainWindow: Rx CAN Error: {error_message}")
        QMessageBox.critical(self, "Lỗi CAN", error_message)
        self.statusBar.showMessage(f"LỖI CAN! {error_message}", 0)
        # on_can_thread_finished sẽ reset UI và nút
        # Có thể reset nút ngay tại đây nếu muốn phản hồi nhanh hơn
        if self.connect_can_btn.isChecked(): self.connect_can_btn.setChecked(False)
        self.connect_can_btn.setText("Kết Nối CAN")
        self.set_can_config_enabled(True)

    # --- Close Event ---
    def closeEvent(self, event):
        """Dọn dẹp khi đóng."""
        print("MainWindow: Close event...")
        reply = QMessageBox.question(self, 'Thoát?', "Đóng ứng dụng?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            self.statusBar.showMessage("Đang đóng...")
            QApplication.processEvents()
            threads = [t for t in (self.can_thread, self.camera_thread) if t and t.isRunning()]
            if threads:
                print(f"  Stopping {len(threads)} thread(s)...")
                for t in threads: t.stop()
                # Đợi tất cả kết thúc
                all_stopped = all(t.wait(1500) for t in threads) # Đợi tối đa 1.5s mỗi thread
                if not all_stopped: print("Warning: Some threads did not stop cleanly.")
            print("Closing application.")
            event.accept()
        else:
            print("Close cancelled.")
            event.ignore()

# ---- Main ----
if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
