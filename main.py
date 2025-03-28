import sys
import os
import datetime
import re
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QComboBox, QLabel, QLineEdit, QFileDialog,
                             QTextEdit, QStatusBar, QMessageBox, QCheckBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QMutex, QMutexLocker
from PyQt5.QtGui import QImage, QPixmap
import cv2
import can
import numpy as np
import time

# ---- Global Settings ----
DEFAULT_SAVE_DIR = os.path.expanduser("~") # Thư mục Home làm mặc định
CAMERA_SCAN_LIMIT = 5 # Số lượng index camera tối đa để quét

# ---- Thread cho Camera ----
class CameraThread(QThread):
    changePixmap = pyqtSignal(QPixmap)
    recordingStartedSignal = pyqtSignal()
    recordingStoppedSignal = pyqtSignal(str) # Gửi đường dẫn file đã lưu
    cameraErrorSignal = pyqtSignal(str)

    def __init__(self, camera_source, save_dir, parent=None):
        super().__init__(parent)
        self.camera_source = camera_source
        self.save_dir = save_dir
        self._running = False
        self._recording = False
        self.video_writer = None
        self.cap = None
        self.error_string_from_can = "UnknownEvent" # Mặc định
        self.mutex = QMutex() # Bảo vệ truy cập vào _recording và video_writer

    def set_save_dir(self, directory):
        self.save_dir = directory

    def run(self):
        self._running = True
        print(f"Attempting to open camera: {self.camera_source}")
        try:
            # Cố gắng chuyển đổi sang int nếu là số (index), không thì giữ nguyên (URL)
            try:
                cam_idx = int(self.camera_source)
                self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW if os.name == 'nt' else None) # Thêm backend cho Windows
            except ValueError:
                self.cap = cv2.VideoCapture(self.camera_source)

            if not self.cap or not self.cap.isOpened():
                raise ConnectionError(f"Không thể mở camera: {self.camera_source}")

            print(f"Camera {self.camera_source} opened successfully.")
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            # Đảm bảo FPS hợp lệ, nếu không lấy được thì đặt mặc định
            if fps <= 0:
                print("Warning: Could not get camera FPS, defaulting to 25.")
                fps = 25 # Giá trị mặc định

            print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps}")

            while self._running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to grab frame.")
                    time.sleep(0.1) # Đợi một chút trước khi thử lại
                    continue

                # --- Xử lý hiển thị ---
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = QPixmap.fromImage(convert_to_qt_format)
                self.changePixmap.emit(p)

                # --- Xử lý ghi video (thread-safe) ---
                with QMutexLocker(self.mutex):
                    if self._recording and self.video_writer:
                        try:
                            self.video_writer.write(frame)
                        except Exception as e:
                            print(f"Error writing frame: {e}")
                            # Có thể dừng ghi ở đây hoặc chỉ log lỗi
                # time.sleep(1 / (fps * 1.1)) # Thêm độ trễ nhỏ nếu cần để giảm CPU, *1.1 để an toàn
                time.sleep(0.01) # Giới hạn tốc độ vòng lặp một chút

        except ConnectionError as e:
             print(f"Camera connection error: {e}")
             self.cameraErrorSignal.emit(str(e))
        except Exception as e:
             print(f"Error in camera thread: {e}")
             self.cameraErrorSignal.emit(f"Lỗi camera: {e}")
        finally:
            print("Camera thread finishing...")
            if self.cap:
                self.cap.release()
                print("Camera released.")
            with QMutexLocker(self.mutex):
                if self.video_writer:
                    print("Releasing video writer...")
                    self.video_writer.release()
                    self.video_writer = None
            self._recording = False # Đảm bảo trạng thái recording là false khi thread kết thúc
            print("Camera thread finished.")


    def stop(self):
        self._running = False
        self.wait() # Đợi thread kết thúc hoàn toàn

    def start_recording(self):
        with QMutexLocker(self.mutex):
            if not self._recording and self.cap and self.cap.isOpened():
                if not self.save_dir or not os.path.isdir(self.save_dir):
                    print("Error: Invalid save directory.")
                    self.cameraErrorSignal.emit("Lỗi: Thư mục lưu không hợp lệ.")
                    return

                frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0: fps = 25 # Default FPS

                # Sử dụng tên tạm thời trước khi có tên lỗi từ CAN
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                self.temp_filename = os.path.join(self.save_dir, f"recording_{timestamp}.mp4")

                # --- Chọn Codec ---
                # Thử 'mp4v', nếu không được có thể thử 'XVID' hoặc khác
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # fourcc = cv2.VideoWriter_fourcc(*'XVID')

                try:
                    print(f"Starting recording to {self.temp_filename} at {fps} FPS, {frame_width}x{frame_height}")
                    self.video_writer = cv2.VideoWriter(self.temp_filename, fourcc, fps, (frame_width, frame_height))
                    if not self.video_writer.isOpened():
                         raise IOError("Could not open VideoWriter")
                    self._recording = True
                    self.recordingStartedSignal.emit()
                    print("Recording started.")
                except Exception as e:
                    print(f"Error starting VideoWriter: {e}")
                    self.cameraErrorSignal.emit(f"Lỗi bắt đầu ghi: {e}")
                    self.video_writer = None
                    self._recording = False


    def stop_recording_and_save(self, error_string="UnknownEvent"):
         final_filepath = ""
         with QMutexLocker(self.mutex):
            if self._recording and self.video_writer:
                print("Stopping recording...")
                self._recording = False
                temp_file_to_rename = self.temp_filename # Lưu lại tên tạm thời
                try:
                    self.video_writer.release()
                    self.video_writer = None
                    print("Video writer released.")

                    # --- Tạo tên file cuối cùng ---
                    # Làm sạch tên lỗi/sự kiện từ CAN
                    safe_error_string = re.sub(r'[\\/*?:"<>|]', "_", error_string) # Thay thế ký tự không hợp lệ
                    if not safe_error_string: safe_error_string = "UnknownEvent" # Đảm bảo không rỗng

                    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
                    final_filename = f"{date_str}_{safe_error_string}.mp4"
                    final_filepath = os.path.join(self.save_dir, final_filename)

                    # Đổi tên file tạm thành file cuối cùng
                    if os.path.exists(temp_file_to_rename):
                         # Xử lý trường hợp file đích đã tồn tại (ví dụ thêm số thứ tự)
                         counter = 1
                         base_name, ext = os.path.splitext(final_filename)
                         while os.path.exists(final_filepath):
                              final_filename = f"{base_name}_{counter}{ext}"
                              final_filepath = os.path.join(self.save_dir, final_filename)
                              counter += 1

                         os.rename(temp_file_to_rename, final_filepath)
                         print(f"Video saved as: {final_filepath}")
                    else:
                         print(f"Warning: Temporary file {temp_file_to_rename} not found for renaming.")
                         final_filepath = "" # Đặt lại nếu không đổi tên được


                except Exception as e:
                    print(f"Error stopping recording or renaming file: {e}")
                    # Không emit lỗi ở đây vì có thể luồng chính đã xử lý
                    final_filepath = "" # Đặt lại nếu có lỗi
                finally:
                     # Dù thành công hay không, báo hiệu đã dừng (có thể kèm đường dẫn hoặc chuỗi rỗng)
                     self.recordingStoppedSignal.emit(final_filepath)
                     print("Recording stopped signal emitted.")

# ---- Thread cho CAN ----
class CanThread(QThread):
    logMessage = pyqtSignal(str)
    startRecordingSignal = pyqtSignal()
    stopRecordingAndSaveSignal = pyqtSignal(str) # Gửi chuỗi lỗi/sự kiện từ payload
    canErrorSignal = pyqtSignal(str)
    connectionStatusSignal = pyqtSignal(bool) # True khi kết nối, False khi ngắt

    def __init__(self, interface, channel, start_id_hex, stop_id_hex, emergency_id_hex=None, parent=None):
        super().__init__(parent)
        self.interface = interface
        self.channel = channel
        self.start_id = None
        self.stop_id = None
        self.emergency_id = None
        self._running = False
        self.bus = None
        self.notifier = None

        # Chuyển đổi ID hex sang int, xử lý lỗi
        try:
            self.start_id = int(start_id_hex, 16) if start_id_hex else None
            self.stop_id = int(stop_id_hex, 16) if stop_id_hex else None
            self.emergency_id = int(emergency_id_hex, 16) if emergency_id_hex else None
            print(f"CAN IDs configured - Start: {hex(self.start_id) if self.start_id is not None else 'None'}, Stop: {hex(self.stop_id) if self.stop_id is not None else 'None'}")
        except ValueError as e:
            raise ValueError(f"Định dạng CAN ID không hợp lệ: {e}")

        if not self.start_id or not self.stop_id:
             raise ValueError("Start Recording ID và Stop Recording ID không được để trống.")


    def run(self):
        self._running = True
        try:
            print(f"Attempting to connect to CAN: interface='{self.interface}', channel='{self.channel}'")
            # --- Khởi tạo CAN Bus ---
            # Cần cấu hình chính xác 'interface' và 'channel' cho phần cứng của bạn
            # Ví dụ cho SocketCAN trên Linux: interface='socketcan', channel='can0'
            # Ví dụ cho PCAN trên Windows: interface='pcan', channel='PCAN_USBBUS1'
            # Ví dụ cho Vector trên Windows: interface='vector', channel=0, app_name='MyCameraApp'
            # **CHỈNH SỬA DÒNG DƯỚI ĐÂY CHO PHÙ HỢP VỚI HỆ THỐNG CỦA BẠN**
            self.bus = can.interface.Bus(interface=self.interface, channel=self.channel, bustype=self.interface)
            # self.bus = can.interface.Bus(bustype='socketcan', channel='can0', bitrate=500000) # Example for SocketCAN
            # self.bus = can.interface.Bus(bustype='pcan', channel='PCAN_USBBUS1', bitrate=500000) # Example for PCAN
            print("CAN bus connected successfully.")
            self.connectionStatusSignal.emit(True)

            # Tạo listener tùy chỉnh
            class MyListener(can.Listener):
                def __init__(self, can_thread_ref):
                    self.can_thread_ref = can_thread_ref

                def on_message_received(self, msg: can.Message):
                    # Log tất cả message (nếu được bật)
                    log_str = f"ID: {msg.arbitration_id:#X} DLC: {msg.dlc} Data: {msg.data.hex().upper()}"
                    self.can_thread_ref.logMessage.emit(log_str)

                    # Xử lý lệnh điều khiển
                    if self.can_thread_ref.start_id is not None and msg.arbitration_id == self.can_thread_ref.start_id:
                        print("Received Start Recording CAN message.")
                        self.can_thread_ref.startRecordingSignal.emit()
                    elif self.can_thread_ref.stop_id is not None and msg.arbitration_id == self.can_thread_ref.stop_id:
                        print("Received Stop Recording CAN message.")
                        try:
                            # Giả định payload là chuỗi UTF-8
                            error_string = msg.data.decode('utf-8', errors='ignore').strip()
                            # Xóa các ký tự null nếu có
                            error_string = error_string.replace('\x00', '')
                            if not error_string: error_string = "ReceivedStopEvent" # Nếu payload rỗng
                            print(f"Extracted error/event string: '{error_string}'")
                        except Exception as e:
                            print(f"Could not decode CAN payload as UTF-8: {e}")
                            error_string = "PayloadDecodeError"
                        self.can_thread_ref.stopRecordingAndSaveSignal.emit(error_string)
                    # Thêm xử lý Emergency Stop nếu cần
                    # elif self.can_thread_ref.emergency_id is not None and msg.arbitration_id == self.can_thread_ref.emergency_id:
                    #     print("Received Emergency Stop CAN message.")
                    #     # Gửi tín hiệu dừng camera khẩn cấp (cần định nghĩa thêm)
                    #     pass

            # Sử dụng Notifier để chạy listener trong nền
            self.notifier = can.Notifier(self.bus, [MyListener(self)])
            print("CAN Notifier started.")

            # Giữ thread chạy cho đến khi bị dừng từ bên ngoài
            while self._running:
                time.sleep(0.5) # Ngủ để giảm tải CPU, notifier vẫn chạy

        except can.CanError as e:
            print(f"CAN Error: {e}")
            self.canErrorSignal.emit(f"Lỗi CAN: {e}")
            self.connectionStatusSignal.emit(False)
        except ValueError as e: # Bắt lỗi từ việc chuyển đổi ID
            print(f"Configuration Error: {e}")
            self.canErrorSignal.emit(str(e))
            self.connectionStatusSignal.emit(False)
        except Exception as e:
            print(f"Unexpected error in CAN thread: {e}")
            self.canErrorSignal.emit(f"Lỗi không xác định: {e}")
            self.connectionStatusSignal.emit(False)
        finally:
            print("CAN thread finishing...")
            if self.notifier:
                self.notifier.stop()
                print("CAN Notifier stopped.")
            if self.bus:
                self.bus.shutdown()
                print("CAN bus shutdown.")
            self._running = False
            self.connectionStatusSignal.emit(False) # Báo đã ngắt kết nối
            print("CAN thread finished.")

    def stop(self):
        self._running = False
        # Không cần wait() ở đây vì vòng lặp chính sẽ tự kết thúc khi _running=False
        # và notifier.stop(), bus.shutdown() sẽ được gọi trong finally


# ---- Giao diện chính ----
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Giám Sát Camera & Điều Khiển CAN")
        self.setGeometry(100, 100, 900, 700) # Tăng kích thước cửa sổ

        # --- Thuộc tính ---
        self.camera_thread = None
        self.can_thread = None
        self.current_save_dir = DEFAULT_SAVE_DIR
        self.is_recording_flag = False # Cờ trạng thái ghi hình
        self.can_log_enabled = True

        # --- Giao diện ---
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget) # Layout chính: Trái (Video), Phải (Điều khiển)

        # -- Khu vực Video (Trái) --
        video_layout = QVBoxLayout()
        self.video_label = QLabel("Chưa kết nối Camera")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480) # Kích thước tối thiểu cho video
        self.video_label.setStyleSheet("border: 1px solid black; background-color: lightgray;")
        video_layout.addWidget(self.video_label)
        main_layout.addLayout(video_layout, 3) # Chiếm 3 phần không gian

        # -- Khu vực Điều khiển (Phải) --
        control_layout = QVBoxLayout()

        # 1. Quản lý Camera
        cam_group = QVBoxLayout()
        cam_group.addWidget(QLabel("1. Quản Lý Camera:"))
        self.cam_combo = QComboBox()
        self.scan_cameras() # Quét khi khởi động
        cam_group.addWidget(self.cam_combo)

        ip_cam_layout = QHBoxLayout()
        self.ip_cam_input = QLineEdit()
        self.ip_cam_input.setPlaceholderText("Nhập URL Camera IP (vd: rtsp://...)")
        ip_cam_layout.addWidget(self.ip_cam_input)
        add_ip_btn = QPushButton("Thêm IP Cam")
        add_ip_btn.clicked.connect(self.add_ip_camera)
        ip_cam_layout.addWidget(add_ip_btn)
        cam_group.addLayout(ip_cam_layout)

        cam_buttons_layout = QHBoxLayout()
        self.start_cam_btn = QPushButton("Bật Camera")
        self.start_cam_btn.clicked.connect(self.start_camera)
        cam_buttons_layout.addWidget(self.start_cam_btn)
        self.stop_cam_btn = QPushButton("Tắt Camera")
        self.stop_cam_btn.clicked.connect(self.stop_camera)
        self.stop_cam_btn.setEnabled(False)
        cam_buttons_layout.addWidget(self.stop_cam_btn)
        # Nút lưu thủ công (chỉ để test)
        # self.manual_rec_btn = QPushButton("Lưu Video (Test)")
        # self.manual_rec_btn.clicked.connect(self.toggle_manual_record)
        # cam_buttons_layout.addWidget(self.manual_rec_btn)
        cam_group.addLayout(cam_buttons_layout)
        control_layout.addLayout(cam_group)

        # 2. Điều khiển Hệ thống
        system_group = QVBoxLayout()
        system_group.addWidget(QLabel("2. Điều Khiển Hệ Thống:"))
        dir_layout = QHBoxLayout()
        self.dir_label = QLineEdit(self.current_save_dir)
        self.dir_label.setReadOnly(True)
        dir_layout.addWidget(self.dir_label)
        select_dir_btn = QPushButton("Chọn Thư Mục")
        select_dir_btn.clicked.connect(self.select_directory)
        dir_layout.addWidget(select_dir_btn)
        system_group.addLayout(dir_layout)
        control_layout.addLayout(system_group)

        # 3. Tích hợp CAN
        can_group = QVBoxLayout()
        can_group.addWidget(QLabel("3. Tích Hợp Giao Tiếp CAN:"))
        can_config_layout = QHBoxLayout()
        self.can_interface_input = QLineEdit("socketcan") # Mặc định cho Linux
        self.can_interface_input.setPlaceholderText("Interface (vd: socketcan, pcan)")
        self.can_channel_input = QLineEdit("can0") # Mặc định cho Linux
        self.can_channel_input.setPlaceholderText("Channel (vd: can0, PCAN_USBBUS1)")
        can_config_layout.addWidget(self.can_interface_input)
        can_config_layout.addWidget(self.can_channel_input)
        can_group.addLayout(can_config_layout)

        can_id_layout = QHBoxLayout()
        self.start_id_input = QLineEdit("100") # ID mặc định ví dụ
        self.start_id_input.setPlaceholderText("ID Bắt Đầu Ghi (Hex)")
        self.stop_id_input = QLineEdit("101") # ID mặc định ví dụ
        self.stop_id_input.setPlaceholderText("ID Dừng & Lưu (Hex)")
        # self.emergency_id_input = QLineEdit() # Tùy chọn
        # self.emergency_id_input.setPlaceholderText("ID Dừng Khẩn Cấp (Hex)")
        can_id_layout.addWidget(self.start_id_input)
        can_id_layout.addWidget(self.stop_id_input)
        # can_id_layout.addWidget(self.emergency_id_input)
        can_group.addLayout(can_id_layout)

        self.connect_can_btn = QPushButton("Kết Nối CAN")
        self.connect_can_btn.setCheckable(True) # Làm nút bật/tắt
        self.connect_can_btn.clicked.connect(self.toggle_can_connection)
        can_group.addWidget(self.connect_can_btn)

        control_layout.addLayout(can_group)

        # 4. Log CAN
        log_group = QVBoxLayout()
        log_header_layout = QHBoxLayout()
        log_header_layout.addWidget(QLabel("4. Log Dữ Liệu CAN:"))
        self.can_log_checkbox = QCheckBox("Bật Log")
        self.can_log_checkbox.setChecked(self.can_log_enabled)
        self.can_log_checkbox.stateChanged.connect(self.toggle_can_logging)
        log_header_layout.addWidget(self.can_log_checkbox)
        log_group.addLayout(log_header_layout)

        self.can_log_display = QTextEdit()
        self.can_log_display.setReadOnly(True)
        log_group.addWidget(self.can_log_display)
        control_layout.addLayout(log_group)

        control_layout.addStretch() # Đẩy mọi thứ lên trên
        main_layout.addLayout(control_layout, 1) # Chiếm 1 phần không gian

        # --- Status Bar ---
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Sẵn sàng")

    # --- Các phương thức xử lý ---

    def scan_cameras(self):
        self.cam_combo.clear()
        available_cameras = []
        print("Scanning for cameras...")
        for i in range(CAMERA_SCAN_LIMIT):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name == 'nt' else None)
            if cap is not None and cap.isOpened():
                cam_name = f"Camera {i}"
                # Cố gắng lấy tên mô tả hơn (có thể không hoạt động trên mọi OS/backend)
                # try:
                #     backend_name = cap.getBackendName()
                #     # Tên cụ thể hơn có thể cần API hệ điều hành
                #     cam_name = f"Camera {i} ({backend_name})"
                # except Exception:
                #     pass
                available_cameras.append((cam_name, str(i))) # Lưu cả tên và index
                print(f"Found: {cam_name}")
                cap.release()
            else:
                # print(f"Index {i} not available or failed to open.")
                if cap: cap.release()
                # Dừng quét sớm nếu không tìm thấy liên tiếp
                # if i > 0 and not available_cameras: # Nếu index 0 không có thì cũng dừng
                 #    break

        if not available_cameras:
            self.cam_combo.addItem("Không tìm thấy camera nào")
            self.cam_combo.setEnabled(False)
        else:
             self.cam_combo.setEnabled(True)
             for name, source in available_cameras:
                 self.cam_combo.addItem(name, userData=source) # Lưu source vào userData

        print("Camera scan finished.")


    def add_ip_camera(self):
        url = self.ip_cam_input.text().strip()
        if url:
            # Kiểm tra cơ bản định dạng URL (có thể cần kiểm tra kỹ hơn)
            if url.startswith("rtsp://") or url.startswith("http://") or url.startswith("https://"):
                 # Kiểm tra xem URL đã tồn tại chưa
                 for i in range(self.cam_combo.count()):
                      if self.cam_combo.itemData(i) == url:
                           QMessageBox.information(self, "Thông báo", f"Camera IP '{url}' đã tồn tại trong danh sách.")
                           return
                 self.cam_combo.addItem(f"IP: {url}", userData=url)
                 self.cam_combo.setCurrentIndex(self.cam_combo.count() - 1) # Chọn camera mới thêm
                 self.ip_cam_input.clear()
                 self.cam_combo.setEnabled(True)
            else:
                 QMessageBox.warning(self, "Lỗi", "URL Camera IP không hợp lệ. Phải bắt đầu bằng rtsp://, http:// hoặc https://")
        else:
             QMessageBox.warning(self, "Lỗi", "Vui lòng nhập URL Camera IP.")


    def start_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            QMessageBox.warning(self, "Lỗi", "Camera đang chạy!")
            return

        selected_index = self.cam_combo.currentIndex()
        if selected_index < 0:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn camera!")
            return

        camera_source = self.cam_combo.itemData(selected_index) # Lấy source từ userData
        if not camera_source:
             QMessageBox.warning(self, "Lỗi", "Không lấy được thông tin camera đã chọn.")
             return

        self.statusBar.showMessage(f"Đang kết nối tới {self.cam_combo.currentText()}...")
        QApplication.processEvents() # Cập nhật giao diện

        self.camera_thread = CameraThread(camera_source, self.current_save_dir)
        self.camera_thread.changePixmap.connect(self.set_image)
        self.camera_thread.recordingStartedSignal.connect(self.on_recording_started)
        self.camera_thread.recordingStoppedSignal.connect(self.on_recording_stopped)
        self.camera_thread.cameraErrorSignal.connect(self.on_camera_error)
        self.camera_thread.finished.connect(self.on_camera_thread_finished) # Xử lý khi thread kết thúc

        self.camera_thread.start()

        self.start_cam_btn.setEnabled(False)
        self.stop_cam_btn.setEnabled(True)
        self.cam_combo.setEnabled(False) # Không cho đổi camera khi đang chạy
        self.ip_cam_input.setEnabled(False) # Tương tự cho IP Cam


    def stop_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.statusBar.showMessage("Đang dừng camera...")
            self.camera_thread.stop() # Gửi tín hiệu dừng và đợi
            # Việc cập nhật UI sẽ được xử lý trong on_camera_thread_finished
        else:
             # Nếu thread không chạy nhưng nút vẫn bật (trường hợp lỗi nào đó)
             self.on_camera_thread_finished() # Reset UI


    def on_camera_thread_finished(self):
        print("Camera thread finished signal received by main window.")
        self.camera_thread = None # Xóa tham chiếu đến thread
        self.video_label.setText("Camera đã tắt")
        self.video_label.setPixmap(QPixmap()) # Xóa hình ảnh cuối cùng
        self.start_cam_btn.setEnabled(True)
        self.stop_cam_btn.setEnabled(False)
        self.cam_combo.setEnabled(True) # Cho phép chọn lại camera
        self.ip_cam_input.setEnabled(True)
        self.is_recording_flag = False # Đảm bảo cờ ghi hình tắt
        self.statusBar.showMessage("Camera đã tắt.")


    def set_image(self, pixmap):
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Chọn Thư Mục Lưu Video", self.current_save_dir)
        if directory:
            self.current_save_dir = directory
            self.dir_label.setText(directory)
            # Cập nhật thư mục lưu cho camera thread nếu đang chạy
            if self.camera_thread and self.camera_thread.isRunning():
                self.camera_thread.set_save_dir(directory)
            print(f"Thư mục lưu được đặt thành: {directory}")


    def toggle_can_connection(self):
         if self.connect_can_btn.isChecked(): # Nếu nút được nhấn (để kết nối)
            if self.can_thread and self.can_thread.isRunning():
                print("CAN thread already running.")
                return # Đã kết nối rồi

            interface = self.can_interface_input.text().strip()
            channel = self.can_channel_input.text().strip()
            start_id_hex = self.start_id_input.text().strip()
            stop_id_hex = self.stop_id_input.text().strip()
            # emergency_id_hex = self.emergency_id_input.text().strip() # Tùy chọn

            if not interface or not channel or not start_id_hex or not stop_id_hex:
                 QMessageBox.warning(self, "Thiếu thông tin", "Vui lòng nhập Interface, Channel, ID Bắt Đầu và ID Dừng.")
                 self.connect_can_btn.setChecked(False) # Bỏ check nút
                 return

            self.statusBar.showMessage("Đang kết nối CAN...")
            try:
                self.can_thread = CanThread(interface, channel, start_id_hex, stop_id_hex) # , emergency_id_hex)
                self.can_thread.logMessage.connect(self.append_can_log)
                self.can_thread.startRecordingSignal.connect(self.handle_start_recording_can)
                self.can_thread.stopRecordingAndSaveSignal.connect(self.handle_stop_recording_can)
                self.can_thread.canErrorSignal.connect(self.on_can_error)
                self.can_thread.connectionStatusSignal.connect(self.on_can_connection_status)
                self.can_thread.finished.connect(self.on_can_thread_finished) # Khi thread kết thúc
                self.can_thread.start()

                # Vô hiệu hóa các input cấu hình khi đang kết nối
                self.can_interface_input.setEnabled(False)
                self.can_channel_input.setEnabled(False)
                self.start_id_input.setEnabled(False)
                self.stop_id_input.setEnabled(False)
                # self.emergency_id_input.setEnabled(False)

            except ValueError as e: # Bắt lỗi khởi tạo CanThread (vd: ID sai định dạng)
                 self.on_can_error(str(e))
                 self.connect_can_btn.setChecked(False) # Bỏ check nút
                 # Kích hoạt lại input nếu lỗi
                 self.can_interface_input.setEnabled(True)
                 self.can_channel_input.setEnabled(True)
                 self.start_id_input.setEnabled(True)
                 self.stop_id_input.setEnabled(True)
            except Exception as e: # Lỗi khác
                 self.on_can_error(f"Lỗi không xác định khi khởi tạo CAN: {e}")
                 self.connect_can_btn.setChecked(False)
                 # Kích hoạt lại input nếu lỗi
                 self.can_interface_input.setEnabled(True)
                 self.can_channel_input.setEnabled(True)
                 self.start_id_input.setEnabled(True)
                 self.stop_id_input.setEnabled(True)

         else: # Nếu nút không được check (để ngắt kết nối)
             if self.can_thread and self.can_thread.isRunning():
                 self.statusBar.showMessage("Đang ngắt kết nối CAN...")
                 self.can_thread.stop()
                 # Việc cập nhật UI sẽ do on_can_thread_finished xử lý
             else:
                 # Nếu thread không chạy nhưng nút vẫn ở trạng thái "Ngắt kết nối"
                 self.on_can_thread_finished() # Reset UI

    def on_can_thread_finished(self):
         print("CAN thread finished signal received by main window.")
         self.can_thread = None
         # self.connect_can_btn.setChecked(False) # Đảm bảo nút ở trạng thái chưa kết nối
         # self.connect_can_btn.setText("Kết Nối CAN")
         # Kích hoạt lại các input cấu hình
         self.can_interface_input.setEnabled(True)
         self.can_channel_input.setEnabled(True)
         self.start_id_input.setEnabled(True)
         self.stop_id_input.setEnabled(True)
         # self.emergency_id_input.setEnabled(True)
         self.statusBar.showMessage("Đã ngắt kết nối CAN.")


    def on_can_connection_status(self, connected):
        if connected:
            self.connect_can_btn.setText("Ngắt Kết Nối CAN")
            self.statusBar.showMessage("Đã kết nối CAN.")
        else:
            # Có thể được gọi khi kết nối thất bại hoặc khi ngắt kết nối thành công
            if self.connect_can_btn.isChecked(): # Nếu vẫn đang ở trạng thái muốn kết nối -> Lỗi
                 self.connect_can_btn.setChecked(False) # Reset nút
            self.connect_can_btn.setText("Kết Nối CAN")
            # Kích hoạt lại cấu hình nếu kết nối thất bại
            self.can_interface_input.setEnabled(True)
            self.can_channel_input.setEnabled(True)
            self.start_id_input.setEnabled(True)
            self.stop_id_input.setEnabled(True)
            # self.emergency_id_input.setEnabled(True)
            # Status message đã được đặt bởi on_can_error hoặc on_can_thread_finished


    def toggle_can_logging(self, state):
        self.can_log_enabled = (state == Qt.Checked)
        print(f"CAN logging {'enabled' if self.can_log_enabled else 'disabled'}")

    def append_can_log(self, message):
        if self.can_log_enabled:
            self.can_log_display.append(f"{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}: {message}")
            # Tự động cuộn xuống dưới
            self.can_log_display.verticalScrollBar().setValue(self.can_log_display.verticalScrollBar().maximum())


    def handle_start_recording_can(self):
        if self.camera_thread and self.camera_thread.isRunning():
             if not self.is_recording_flag: # Chỉ bắt đầu nếu chưa ghi
                 print("Main: Received start recording signal from CAN")
                 self.is_recording_flag = True # Đặt cờ trước khi gọi thread
                 self.camera_thread.start_recording()
                 # Status sẽ được cập nhật bởi signal từ camera_thread
             else:
                 print("Main: Received start recording signal, but already recording.")
        else:
            print("Main: Received start recording signal, but camera is not running.")
            # Có thể thêm thông báo lỗi ở đây nếu muốn


    def handle_stop_recording_can(self, error_string):
        if self.camera_thread and self.camera_thread.isRunning():
             if self.is_recording_flag: # Chỉ dừng nếu đang ghi
                 print(f"Main: Received stop recording signal from CAN with payload: '{error_string}'")
                 self.is_recording_flag = False # Đặt cờ trước khi gọi thread
                 self.camera_thread.stop_recording_and_save(error_string)
                 # Status sẽ được cập nhật bởi signal từ camera_thread
             else:
                  print("Main: Received stop recording signal, but not currently recording.")
        else:
            print("Main: Received stop recording signal, but camera is not running.")


    def on_recording_started(self):
         self.is_recording_flag = True # Đảm bảo cờ đúng trạng thái
         self.statusBar.showMessage("Đang ghi hình...")
         print("Main: Recording started confirmation received.")

    def on_recording_stopped(self, saved_filepath):
         self.is_recording_flag = False # Đảm bảo cờ đúng trạng thái
         if saved_filepath:
              self.statusBar.showMessage(f"Đã dừng ghi hình. Lưu tại: {saved_filepath}")
              print(f"Main: Recording stopped confirmation received. Saved to {saved_filepath}")
         else:
              # Trường hợp dừng nhưng không lưu được file (lỗi hoặc không có gì để lưu)
              self.statusBar.showMessage("Đã dừng ghi hình (Lỗi lưu file hoặc không có dữ liệu).")
              print("Main: Recording stopped confirmation received, but file saving failed or no data.")


    def on_camera_error(self, error_message):
        QMessageBox.critical(self, "Lỗi Camera", error_message)
        self.statusBar.showMessage(f"Lỗi Camera: {error_message}")
        # Dừng thread camera nếu nó vẫn đang chạy (ví dụ lỗi khi đang ghi)
        if self.camera_thread and self.camera_thread.isRunning():
             self.camera_thread.stop() # Sẽ gọi on_camera_thread_finished để reset UI
        else:
             # Nếu lỗi xảy ra trước khi thread chạy hoặc sau khi đã dừng
             self.on_camera_thread_finished() # Reset UI


    def on_can_error(self, error_message):
        QMessageBox.critical(self, "Lỗi CAN", error_message)
        self.statusBar.showMessage(f"Lỗi CAN: {error_message}")
        # Không cần dừng thread ở đây vì CanThread tự xử lý shutdown và phát signal finished
        # Chỉ cần đảm bảo UI được reset
        self.on_can_connection_status(False) # Cập nhật trạng thái nút và input


    # --- Đóng ứng dụng ---
    def closeEvent(self, event):
        print("Close event triggered.")
        # Dừng các thread trước khi thoát
        if self.camera_thread and self.camera_thread.isRunning():
            print("Stopping camera thread before closing...")
            self.camera_thread.stop()
        if self.can_thread and self.can_thread.isRunning():
            print("Stopping CAN thread before closing...")
            self.can_thread.stop()
        print("Proceeding with closing.")
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
