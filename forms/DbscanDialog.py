from qgis.core import *
from qgis.gui import QgsRubberBand, QgsMapLayerComboBox
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QSlider, QLabel, QPushButton, QHBoxLayout, QSpinBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor, QPixmap
import math, colorsys, matplotlib.pyplot as plt, tempfile

class Config:
    min_points = 4
    eps = None
    cluster_field = 'CLUSTER_ID'
    output_memory_id = 'memory:dbscan_result'
    timer_interval = 200
    alpha_step = 10
    initial_alpha = 100
    marker_size = '1.5'
    eps_min = 1.0
    eps_max = 10.0

# 다이얼로그 정의
class ParameterControlDialog(QDialog):
    def __init__(self, layer: QgsVectorLayer, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DBSCAN Parameters")
        self.setMinimumWidth(400)
        self.animator = None
        self.layer = layer

        Config.eps = self.compute_knn_kth_distance(Config.min_points)
        self.compute_knn_kth_distance(Config.min_points, eps_for_ratio=Config.eps)

        layout = QVBoxLayout()

        self.minpts_label = QLabel(f"MinPts: {Config.min_points}")
        layout.addWidget(self.minpts_label)
        self.minpts_spin = QSpinBox()
        self.minpts_spin.setMinimum(1)
        self.minpts_spin.setMaximum(100)
        self.minpts_spin.setValue(Config.min_points)	
        self.minpts_spin.valueChanged.connect(self.update_minpts)
        layout.addWidget(self.minpts_spin)

        eps_text = f"{Config.eps:.2f}" if Config.eps is not None else "Not set"
        self.eps_label = QLabel(f"Epsilon: {eps_text}")
        layout.addWidget(self.eps_label)
        self.eps_slider = QSlider(Qt.Horizontal)
        self.eps_slider.setMinimum(int(Config.eps_min * 10))
        self.eps_slider.setMaximum(int(Config.eps_max * 10))
        self.eps_slider.setValue(int(Config.eps * 10) if Config.eps is not None else 10)
        self.eps_slider.valueChanged.connect(self.update_eps)
        layout.addWidget(self.eps_slider)

        self.interval_label = QLabel("Interval: 200 ms")
        layout.addWidget(self.interval_label)
        self.interval_slider = QSlider(Qt.Horizontal)
        self.interval_slider.setMinimum(10)
        self.interval_slider.setMaximum(1000)
        self.interval_slider.setValue(Config.timer_interval)
        self.interval_slider.valueChanged.connect(self.update_speed)
        layout.addWidget(self.interval_slider)

        self.fill_alpha_label = QLabel("Fill Opacity: 2%")
        layout.addWidget(self.fill_alpha_label)
        self.fill_alpha_slider = QSlider(Qt.Horizontal)
        self.fill_alpha_slider.setMinimum(0)
        self.fill_alpha_slider.setMaximum(255)
        self.fill_alpha_slider.setValue(5)
        self.fill_alpha_slider.valueChanged.connect(self.update_alpha)
        layout.addWidget(self.fill_alpha_slider)

        btns = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play")
        self.pause_btn = QPushButton("⏸ Pause")
        self.stop_btn = QPushButton("■ Stop")
        self.play_btn.clicked.connect(self.play)
        self.pause_btn.clicked.connect(self.pause)
        self.stop_btn.clicked.connect(self.stop)
        btns.addWidget(self.play_btn)
        btns.addWidget(self.pause_btn)
        btns.addWidget(self.stop_btn)
        layout.addLayout(btns)

        self.plot_label = QLabel()
        self.refresh_plot()
        layout.addWidget(self.plot_label)

        self.setLayout(layout)

    def update_minpts(self, value):
        Config.min_points = value
        self.minpts_label.setText(f"MinPts: {value}")
        Config.eps = self.compute_knn_kth_distance(Config.min_points)
        self.eps_slider.setMinimum(int(Config.eps_min * 10))
        self.eps_slider.setMaximum(int(Config.eps_max * 10))
        self.eps_slider.setValue(int(Config.eps * 10))
        self.eps_label.setText(f"Epsilon: {Config.eps:.2f}")
        self.refresh_plot()

    def update_eps(self, value):
        Config.eps = value / 10.0
        self.eps_label.setText(f"Epsilon: {Config.eps:.2f}")
        self.refresh_plot()

    def refresh_plot(self):
        self.plot_label.setPixmap(self.get_elbow_pixmap())

    def update_speed(self, value):
        Config.timer_interval = value
        self.interval_label.setText(f"Interval: {value} ms")
        if self.animator:
            self.animator.timer.setInterval(value)

    def update_alpha(self, value):
        percent = round(value / 255 * 100)
        self.fill_alpha_label.setText(f"Fill Opacity: {percent}%")

    def play(self):
        layer = self.layer
        if not self.animator or self.animator.layer != layer:
            self.animator = DBSCANAnimator(layer)
            self.animator.run_dbscan_and_prepare()
        else:
            self.animator.resume()

    def pause(self):
        if self.animator:
            self.animator.pause()

    def stop(self):
        if self.animator:
            self.animator.stop()

    def generate_color(self, index, total):
        h = (index / max(1, total)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        return QColor(int(r * 255), int(g * 255), int(b * 255))

    def compute_knn_kth_distance(self, k, eps_for_ratio=None):
        self.sorted_knn_dists = []
        self.below_eps_ratio = 0.0
        feats = list(self.layer.getFeatures())
        points = [f.geometry().asPoint() for f in feats]
        kth_dists = []
        for i, pt1 in enumerate(points):
            dist = sorted([math.hypot(pt1.x() - pt2.x(), pt1.y() - pt2.y()) for j, pt2 in enumerate(points) if i != j])
            if len(dist) >= k:
                kth_dists.append(dist[k - 1])
        self.sorted_knn_dists = sorted(kth_dists, reverse=True)
        if eps_for_ratio is not None:
            self.below_eps_ratio = sum(1 for d in kth_dists if d <= eps_for_ratio) / len(kth_dists) if kth_dists else 0
        if kth_dists:
            Config.eps_min = min(kth_dists)
            Config.eps_max = max(kth_dists)
        return sum(kth_dists) / len(kth_dists) if kth_dists else 10.0

    def update_elbow_plot(self, current_eps):
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.plot(range(1, len(self.sorted_knn_dists)+1), self.sorted_knn_dists, marker='o', color='white', markerfacecolor='blue', markeredgecolor='black', markersize=0.5)
        ax.axhline(y=current_eps, color='r', linestyle='--', label=f"epsilon = {current_eps:.2f}")
        ax.set_title("k-NN Distance Distribution")
        ax.set_xlabel("Points")
        ax.set_ylabel(f"{Config.min_points}-NN Distance")
        ax.legend()
        y_pos = current_eps * 1.02
        ax.text(len(self.sorted_knn_dists) * 0.95, y_pos, f"{self.below_eps_ratio*100:.1f}% below epsilon", va='bottom', ha='right', color='blue')
        plt.tight_layout()
        fd, elbow_img_path = tempfile.mkstemp(suffix=".png")
        plt.savefig(elbow_img_path)
        plt.close()
        return elbow_img_path

    def get_elbow_pixmap(self):
        self.compute_knn_kth_distance(Config.min_points, eps_for_ratio=Config.eps)
        elbow_img_path = self.update_elbow_plot(Config.eps)
        return QPixmap(elbow_img_path)
