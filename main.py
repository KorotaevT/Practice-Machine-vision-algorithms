import sys
import cv2
import time
from concurrent.futures import ThreadPoolExecutor
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QBrush, QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QWidget, QFileDialog, \
    QGroupBox, QHBoxLayout, QScrollArea, QSpacerItem, QSizePolicy, QCheckBox

Conf_threshold = 0.4
NMS_threshold = 0.4
COLORS = [(0, 0, 255), (255, 0, 0), (255, 255, 0),
          (255, 0, 255), (0, 255, 255)]

# Rest of the code remains the same


class_name = []
with open('classes.names', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)


class SettingsWindow(QMainWindow):
    settings_applied = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Настройки")
        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        title_label = QLabel("Выбор активных объектов:")
        layout.addWidget(title_label)

        self.object_list = ["person", "cell phone", "cat"]
        self.checkboxes = []

        for obj in self.object_list:
            checkbox = QCheckBox(obj)
            checkbox.setChecked(obj in MainWindow.selected_objects)
            layout.addWidget(checkbox)
            self.checkboxes.append(checkbox)

        apply_button = QPushButton("Применить")
        apply_button.clicked.connect(self.apply_settings)
        layout.addWidget(apply_button)

    def apply_settings(self):
        selected_objects = []
        for checkbox in self.checkboxes:
            if checkbox.isChecked():
                selected_objects.append(checkbox.text())
        MainWindow.selected_objects = selected_objects
        self.hide()

    def open_settings_window(self):
        settings_window = SettingsWindow()
        settings_window.show()


class MainWindow(QMainWindow):
    selected_objects = ["person", "cell phone", "cat"]  # Список выбранных объектов

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Программа обработки видео")
        self.setup_ui()
        self.video_path = None
        self.cap = None
        self.starting_time = 0
        self.frame_counter = 0
        self.thread_executor = ThreadPoolExecutor()
        self.prev_classes = None
        self.prev_scores = None
        self.prev_boxes = None
        self.settings_window = SettingsWindow()
        self.paused = False

    def setup_ui(self):
        self.setFixedSize(1400, 600)
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QHBoxLayout(central_widget)

        button_widget = QWidget()
        layout.addWidget(button_widget)

        button_layout = QVBoxLayout(button_widget)
        button_layout.setSpacing(10)
        button_layout.setContentsMargins(0, 0, 0, 0)

        spacer_top = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        button_layout.addItem(spacer_top)

        video_button = QPushButton("Выбрать видео")
        video_button.clicked.connect(self.video_selection)
        button_layout.addWidget(video_button)

        camera_button = QPushButton("Выбрать изображение с камеры")
        camera_button.clicked.connect(self.camera_selection)
        button_layout.addWidget(camera_button)

        pause_button = QPushButton("Пауза")
        pause_button.clicked.connect(self.pause_video)
        button_layout.addWidget(pause_button)

        settings_button = QPushButton("Настройки")
        settings_button.clicked.connect(self.open_settings_window)
        button_layout.addWidget(settings_button)

        termination_button = QPushButton("Закрыть программу")
        termination_button.clicked.connect(self.program_termination)
        button_layout.addWidget(termination_button)

        spacer_bottom = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        button_layout.addItem(spacer_bottom)

        self.video_label = QLabel()
        self.video_label.setFixedSize(800, 600)
        layout.addWidget(self.video_label)

        panel_group_box = QGroupBox()
        panel_group_box.setStyleSheet("background-color: rgba(0, 100, 0, 100);")
        panel_layout = QVBoxLayout(panel_group_box)
        panel_layout.setSpacing(0)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setAlignment(Qt.AlignTop)
        layout.addWidget(panel_group_box)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("background-color: rgba(255, 255, 255, 100);")
        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignTop)
        self.info_label.setFixedSize(330, 600)
        scroll_area.setWidget(self.info_label)
        panel_layout.addWidget(scroll_area)

        self.settings_window = SettingsWindow()

        self.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(167, "
                           "206, 97, 255), stop:1 rgba(61, 165, 77, 255));")
        button_widget.setStyleSheet("background-color: rgba(255, 255, 255, 100);")

    def open_settings_window(self):
        self.settings_window = SettingsWindow()
        self.settings_window.selected_objects = MainWindow.selected_objects
        self.settings_window.show()

    def pause_video(self):
        if self.cap is not None:
            if self.paused:
                self.paused = False
                self.starting_time = time.time() - self.frame_counter / self.cap.get(cv2.CAP_PROP_FPS)
                self.sender().setText("Пауза")
            else:
                self.paused = True
                self.sender().setText("Возобновить")

    def process_frame(self, frame):
        if self.frame_counter % 5 == 0 or self.frame_counter < 5:
            classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
            self.prev_classes = classes
            self.prev_scores = scores
            self.prev_boxes = boxes

        if self.prev_classes is not None:
            info_text = ""
            for (classid, score, box) in zip(self.prev_classes, self.prev_scores, self.prev_boxes):
                class_index = int(classid)
                if class_name[
                    class_index] in self.selected_objects:  # Проверка наличия класса в списке выбранных объектов
                    color = COLORS[class_index % len(COLORS)]
                    label = "%s : %f" % (class_name[class_index], score)
                    cv2.rectangle(frame, box, color, 1)
                    cv2.putText(frame, label, (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
                    info_text += f"Type: {class_name[class_index]}, Coords: {box}\n"

            self.info_label.setText(info_text)

        return frame

    def display_frame(self, frame):
        ending_time = time.time() - self.starting_time
        fps = self.frame_counter / ending_time

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)

        pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio)

        self.video_label.setPixmap(pixmap)

        self.info_label.setStyleSheet("background-color: rgba(0, 255, 0, 100);")

    def video_selection(self):
        self.video_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите видео", "", "Video Files (*.mp4 *.avi)"
        )

        if self.video_path:
            if self.cap:
                self.cap.release()

            self.cap = cv2.VideoCapture(self.video_path)
            self.starting_time = time.time()
            self.frame_counter = 0

            self.process_video()

    def camera_selection(self):
        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(1)
        self.starting_time = time.time()
        self.frame_counter = 0

        self.process_video()

    def process_video(self):
        if self.paused:
            QTimer.singleShot(1, self.process_video)
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        self.frame_counter += 1
        frame = cv2.resize(frame, (800, 600))

        future = self.thread_executor.submit(self.process_frame, frame)
        processed_frame = future.result()

        self.display_frame(processed_frame)

        QTimer.singleShot(1, self.process_video)

    def program_termination(self):
        if self.cap:
            self.cap.release()

        self.thread_executor.shutdown()
        self.close()



if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
