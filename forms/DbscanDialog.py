from qgis.core import *
from qgis import processing
from qgis.utils import iface
from qgis.gui import QgsRubberBand, QgsMapLayerComboBox
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QSlider, QLabel, QPushButton, QHBoxLayout, QSpinBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor, QPixmap
import math, colorsys, matplotlib.pyplot as plt, tempfile
from PyQt5.QtCore import QTimer


class ParameterControlDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.iface = iface
        iface.messageBar().clearWidgets()

        self.setWindowTitle("DBSCAN Parameters")
        self.setMinimumWidth(400)

        self.animator = None
        self.layer = None
        self.layer_id = None

        self.min_points = 4
        self.cluster_field = 'CLUSTER_ID'
        self.output_memory_id = 'memory:dbscan_result'
        self.timer_interval = 200
        self.alpha_step = 10
        self.initial_alpha = int(0.1 * 255)
        self.fill_alpha = 2
        self.marker_size = '1.5'
        self.eps_min = 1.0
        self.eps_max = 10.0
        self.eps = None
        
        layout = QVBoxLayout()
        self.control_widgets = []

        # 레이어 콤보박스
        self.layer_combo = QgsMapLayerComboBox()
        self.layer_combo.setFilters(QgsMapLayerProxyModel.PointLayer)

        self.layer_combo.layerChanged.connect(self.on_layer_changed)
        layout.addWidget(self.layer_combo)
        
        # MinPts
        self.minpts_label = QLabel(f"MinPts: {self.min_points}")
        layout.addWidget(self.minpts_label)
        self.control_widgets.append(self.minpts_label)

        self.minpts_spin = QSpinBox()
        self.minpts_spin.setMinimum(1)
        self.minpts_spin.setMaximum(100)
        self.minpts_spin.setValue(self.min_points)	
        self.minpts_spin.valueChanged.connect(self.update_minpts)
        layout.addWidget(self.minpts_spin)
        self.control_widgets.append(self.minpts_spin)

        # Epsilon
        self.eps_label = QLabel("Epsilon: Not set")
        layout.addWidget(self.eps_label)
        self.control_widgets.append(self.eps_label)
        
        self.eps_slider = QSlider(Qt.Horizontal)
        self.eps_slider.setMinimum(int(self.eps_min * 10))
        self.eps_slider.setMaximum(int(self.eps_max * 10))
        self.eps_slider.valueChanged.connect(self.update_eps)
        layout.addWidget(self.eps_slider)
        self.control_widgets.append(self.eps_slider)

        # Interval
        self.interval_label = QLabel(f"Interval: {self.timer_interval} ms")
        layout.addWidget(self.interval_label)
        self.control_widgets.append(self.interval_label)

        self.interval_slider = QSlider(Qt.Horizontal)
        self.interval_slider.setMinimum(10)
        self.interval_slider.setMaximum(1000)
        self.interval_slider.setValue(self.timer_interval)
        self.interval_slider.valueChanged.connect(self.update_speed)
        layout.addWidget(self.interval_slider)
        self.control_widgets.append(self.interval_slider)

        # Fill Opacity
        self.fill_alpha_label = QLabel(f"Fill Opacity: {round(self.initial_alpha / 255 * 100)}%")
        layout.addWidget(self.fill_alpha_label)
        self.control_widgets.append(self.fill_alpha_label)

        self.fill_alpha_slider = QSlider(Qt.Horizontal)
        self.fill_alpha_slider.setMinimum(0)
        self.fill_alpha_slider.setMaximum(255)
        self.fill_alpha_slider.setValue(self.initial_alpha)
        self.fill_alpha_slider.valueChanged.connect(self.update_alpha)
        layout.addWidget(self.fill_alpha_slider)
        self.control_widgets.append(self.fill_alpha_slider)

        # Buttons
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
        self.control_widgets.extend([self.play_btn, self.pause_btn, self.stop_btn])

        self.show_result_btn = QPushButton("DBSCAN Result")
        self.show_result_btn.clicked.connect(self.show_result)
        btns.addWidget(self.show_result_btn)


        # Elbow Plot
        self.plot_label = QLabel()
        layout.addWidget(self.plot_label)
        self.control_widgets.append(self.plot_label)

        self.setLayout(layout)
        
        # 초기 상태: 포인트 레이어 유무 확인
        point_layers = [
            l for l in QgsProject.instance().mapLayers().values()
            if isinstance(l, QgsVectorLayer) and l.geometryType() == QgsWkbTypes.PointGeometry
        ]
        if point_layers:
            self.layer_combo.setLayer(point_layers[0])
            self.on_layer_changed(point_layers[0])
            for w in self.control_widgets:
                w.setEnabled(True)
        else:
            for w in self.control_widgets:
                w.setEnabled(False)

    def on_layer_changed(self, layer):
        if isinstance(layer, QgsVectorLayer) and layer.geometryType() == QgsWkbTypes.PointGeometry:
            self.layer = layer
            self.layer_id = layer.id()
            
            # MinPts 초기화
            self.min_points = 4
            self.minpts_spin.setValue(self.min_points)
            
            # Epsilon 계산 및 UI 반영
            self.eps = self.compute_knn_kth_distance(self.min_points, eps_for_ratio=True)
            eps_text = f"{self.eps:.2f}"
            self.eps_label.setText(f"Epsilon: {eps_text}")
            self.eps_slider.setMinimum(int(self.eps_min * 10))
            self.eps_slider.setMaximum(int(self.eps_max * 10))
            self.eps_slider.setValue(int(self.eps * 10))
            
            # 나머지 설정도 초기화
            self.interval_slider.setValue(self.timer_interval)
            self.fill_alpha_slider.setValue(self.initial_alpha)

            # 애니메이터 제거
            self.animator = None

            # elbow plot 초기화
            self.refresh_plot()

            # 모든 위젯 활성화
            self.minpts_spin.setEnabled(True)
            self.eps_slider.setEnabled(True)
            self.interval_slider.setEnabled(True)
            self.fill_alpha_slider.setEnabled(True)
            self.play_btn.setEnabled(True)
            self.pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)

        else:
            # 포인트 레이어가 아닌 경우 비활성화
            self.layer = None
            self.layer_id = None
            self.minpts_spin.setEnabled(False)
            self.eps_slider.setEnabled(False)
            self.interval_slider.setEnabled(False)
            self.fill_alpha_slider.setEnabled(False)
            self.play_btn.setEnabled(False)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.eps_label.setText("Epsilon: Not set")
            self.minpts_label.setText("MinPts: -")
            self.plot_label.clear()

    def update_minpts(self, value):
        self.min_points = value
        self.minpts_label.setText(f"MinPts: {value}")
        self.eps = self.compute_knn_kth_distance(self.min_points)
        self.eps_slider.setMinimum(int(self.eps_min * 10))
        self.eps_slider.setMaximum(int(self.eps_max * 10))
        self.eps_slider.setValue(int(self.eps * 10))
        self.eps_label.setText(f"Epsilon: {self.eps:.2f}")
        self.refresh_plot()

    def update_eps(self, value):
        self.eps = value / 10.0
        self.eps_label.setText(f"Epsilon: {self.eps:.2f}")
        self.refresh_plot()

    def refresh_plot(self):
        self.plot_label.setPixmap(self.get_elbow_pixmap())

    def update_speed(self, value):
        self.timer_interval = value
        self.interval_label.setText(f"Interval: {value} ms")
        if self.animator:
            self.animator.timer.setInterval(value)

    def update_alpha(self, value):
        percent = round(value / 255 * 100)
        self.fill_alpha_label.setText(f"Fill Opacity: {percent}%")
        self.fill_alpha = value

    def play(self):
        if not self.layer:
            QMessageBox.warning(self, "No Layer", "포인트 벡터 레이어를 선택해야 합니다.")
            return

        layer = self.layer
        if not self.animator or self.animator.layer != layer or self.animator.i == 0:
            self.animator = DBSCANAnimator(self)
            self.animator.run_dbscan_and_prepare()
            # 비활성화
            self.minpts_spin.setEnabled(False)
            self.eps_slider.setEnabled(False)

        else:
            self.animator.resume()
            self.minpts_spin.setEnabled(False)
            self.eps_slider.setEnabled(False)

    def pause(self):
        if not self.layer:
            QMessageBox.warning(self, "No Layer", "레이어를 선택해야 합니다.")
            return
        if self.animator:
            self.animator.pause()

    def stop(self):
        if not self.layer:
            QMessageBox.warning(self, "No Layer", "레이어를 선택해야 합니다.")
            return
        if self.animator:
            self.animator.stop()
        # 활성화
        self.minpts_spin.setEnabled(True)
        self.eps_slider.setEnabled(True)

    def show_result(self):
        if not self.layer:
            QMessageBox.warning(self, "Error", "레이어가 선택되지 않았습니다.")
            return

        result = processing.run("native:dbscanclustering", {
            'INPUT': self.layer,
            'EPS': self.eps,
            'MIN_POINTS': self.min_points,
            'OUTPUT': 'memory:dbscan_result'
        })

        clustered_layer = result['OUTPUT']
        clustered_layer.setName("DBSCAN_Result")

        cluster_field = self.cluster_field
        cluster_idx = clustered_layer.fields().indexFromName(cluster_field)

        features = list(clustered_layer.getFeatures())
        cluster_ids = sorted(set(f[cluster_field] for f in features if f[cluster_field] not in (None, -1)))

        color_map = {cid: self.generate_color(i, len(cluster_ids)) for i, cid in enumerate(cluster_ids)}

        categories = [
            QgsRendererCategory(cid, QgsMarkerSymbol.createSimple({'name': 'circle', 'color': color.name(), 'size': self.marker_size}), f"Cluster {cid}")
            for cid, color in color_map.items()
        ]
        categories.append(QgsRendererCategory(-1, QgsMarkerSymbol.createSimple({'name': 'circle', 'color': 'black', 'size': self.marker_size}), "Noise"))

        renderer = QgsCategorizedSymbolRenderer(cluster_field, categories)
        clustered_layer.setRenderer(renderer)

        QgsProject.instance().addMapLayer(clustered_layer)

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
            self.eps_min = min(kth_dists)
            self.eps_max = max(kth_dists)
        return sum(kth_dists) / len(kth_dists) if kth_dists else 10.0

    def update_elbow_plot(self, current_eps):
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.plot(range(1, len(self.sorted_knn_dists)+1), self.sorted_knn_dists, marker='o', color='white', markerfacecolor='blue', markeredgecolor='black', markersize=0.5)
        ax.axhline(y=current_eps, color='r', linestyle='--', label=f"epsilon = {current_eps:.2f}")
        ax.set_title("k-NN Distance Distribution")
        ax.set_xlabel("Points")
        ax.set_ylabel(f"{self.min_points}-NN Distance")
        ax.legend()
        y_pos = current_eps * 1.02
        ax.text(len(self.sorted_knn_dists) * 0.95, y_pos, f"{self.below_eps_ratio*100:.1f}% below epsilon", va='bottom', ha='right', color='blue')
        plt.tight_layout()
        fd, elbow_img_path = tempfile.mkstemp(suffix=".png")
        plt.savefig(elbow_img_path)
        plt.close()
        return elbow_img_path

    def get_elbow_pixmap(self):
        self.compute_knn_kth_distance(self.min_points, eps_for_ratio=self.eps)
        elbow_img_path = self.update_elbow_plot(self.eps)
        return QPixmap(elbow_img_path)

class DBSCANAnimator:
    def __init__(self, dialog):
        self.dialog = dialog
        self.layer = dialog.layer
        self.layer_id = dialog.layer_id
        self.iface = dialog.iface

        self.clustered_layer = None
        self.features = []
        self.points = []
        self.visit_order = []
        self.distance_matrix = []
        self.color_map = {}
        self.cluster_idx = -1
        self.transform = None
        self.center_point = None
        self.i = 0

        self.rubber = QgsRubberBand(self.iface.mapCanvas(), QgsWkbTypes.PolygonGeometry)
        self.rubber.setFillColor(Qt.transparent)
        self.rubber.setWidth(2)
        self.rubber.setLineStyle(Qt.SolidLine)
        self.previous_rubbers = []

        self.timer = QTimer()
        self.timer.setInterval(self.dialog.timer_interval)
        self.timer.timeout.connect(self.step)

        self.alpha_timer = QTimer()
        self.alpha_timer.timeout.connect(self.grow_circle)
        self.rubber_alpha = self.dialog.initial_alpha

    def generate_color(self, index, total):
        h = (index / max(1, total)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        return QColor(int(r * 255), int(g * 255), int(b * 255))

    def run_dbscan_and_prepare(self):
        self.rubber.reset(QgsWkbTypes.PolygonGeometry)
        for r in self.previous_rubbers:
            r.reset(QgsWkbTypes.PolygonGeometry)
        self.previous_rubbers.clear()
        for item in self.iface.mapCanvas().scene().items():
            if isinstance(item, QgsRubberBand):
                item.reset(QgsWkbTypes.PolygonGeometry)
        layer = QgsProject.instance().mapLayer(self.layer_id)

        result = processing.run("native:dbscanclustering", {
            'INPUT': layer,
            'EPS': self.dialog.eps,
            'MIN_POINTS': self.dialog.min_points,
            'OUTPUT': 'memory:dbscan_result'
        })

        self.clustered_layer = result['OUTPUT']
        # QgsProject.instance().addMapLayer(self.clustered_layer)
        self.clustered_layer.setName("DBSCAN_Result")

        if self.dialog.cluster_field not in [f.name() for f in self.clustered_layer.fields()]:
            raise Exception(f"'{self.dialog.cluster_field}' 필드가 없습니다.")

        self.features = list(self.clustered_layer.getFeatures())
        self.cluster_idx = self.clustered_layer.fields().indexFromName(self.dialog.cluster_field)

        cluster_ids = sorted(set(f[self.dialog.cluster_field] for f in self.features if f[self.dialog.cluster_field] is not None))
        self.color_map.clear()
        self.color_map.update({cid: self.generate_color(i, len(cluster_ids)) for i, cid in enumerate(cluster_ids)})

        categories = [
            QgsRendererCategory(cid, QgsMarkerSymbol.createSimple({
                'name': 'circle',
                'color': color.name(),
                'size': self.dialog.marker_size
            }), f"Cluster {cid}")
            for cid, color in self.color_map.items()
        ]

        # NULL → Noise로 설정
        noise_symbol = QgsMarkerSymbol.createSimple({
            'name': 'circle',
            'color': 'gray',
            'size': self.dialog.marker_size
        })
        categories.append(QgsRendererCategory(None, noise_symbol, "Noise"))

        renderer = QgsCategorizedSymbolRenderer(self.dialog.cluster_field, categories)
        self.clustered_layer.setRenderer(renderer)
        self.clustered_layer.triggerRepaint()

        self.points = [f.geometry().asPoint() for f in self.features]
        self.distance_matrix = [[math.hypot(p1.x() - p2.x(), p1.y() - p2.y()) for p2 in self.points] for p1 in self.points]
        self.visit_order = self._compute_visit_order()
        self.transform = QgsCoordinateTransform(self.clustered_layer.crs(), self.iface.mapCanvas().mapSettings().destinationCrs(), QgsProject.instance())
        self.center_point = self.clustered_layer.extent().center()
        self.iface.mapCanvas().setExtent(self.clustered_layer.extent())
        self.iface.mapCanvas().refresh()
        self.i = 0
        self.timer.start()

    def _compute_visit_order(self):
        visited = {0}
        order = [0]
        while len(order) < len(self.points):
            last = order[-1]
            candidates = [(j, self.distance_matrix[last][j]) for j in range(len(self.points)) if j not in visited]
            if candidates:
                next_idx = min(candidates, key=lambda x: x[1])[0]
                order.append(next_idx)
                visited.add(next_idx)
        return order

    def step(self):
        if self.i >= len(self.visit_order):
            self.timer.stop()
            self.alpha_timer.stop()
            self.rubber.reset(QgsWkbTypes.PolygonGeometry)  # 원형 강조 효과 제거
            for r in self.previous_rubbers:
                r.reset(QgsWkbTypes.PolygonGeometry)        # 모든 채운 rubber 제거
            self.previous_rubbers.clear()
            self.iface.mapCanvas().refresh()
            self.i = 0

            # 다이얼로그 입력 위젯 다시 활성화
            self.dialog.minpts_spin.setEnabled(True)
            self.dialog.eps_slider.setEnabled(True)
            
            # 메모리 레이어 삭제
            if self.clustered_layer:
                try:
                    # 프로젝트에 추가된 게 아니면 이건 필요 없지만, 혹시라도 등록되었으면 제거
                    QgsProject.instance().removeMapLayer(self.clustered_layer.id())
                except:
                    pass
                self.clustered_layer = None

            return


        if self.i > 0:
            prev_idx = self.visit_order[self.i - 1]
            pt_canvas = self.transform.transform(self.points[prev_idx])
            cluster_id = self.features[prev_idx][self.dialog.cluster_field]
            color = self.color_map.get(cluster_id, QColor(0, 0, 0))
            color.setAlpha(self.dialog.fill_alpha)
            fill = QgsRubberBand(self.iface.mapCanvas(), QgsWkbTypes.PolygonGeometry)
            fill.setWidth(0)
            fill.setColor(Qt.transparent)
            fill.setFillColor(color)
            for angle in range(0, 360, 10):
                rad = math.radians(angle)
                x = pt_canvas.x() + self.dialog.eps * math.cos(rad)
                y = pt_canvas.y() + self.dialog.eps * math.sin(rad)
                fill.addPoint(QgsPointXY(x, y))
            fill.closePoints()
            self.previous_rubbers.append(fill)

        idx = self.visit_order[self.i]
        fid = self.features[idx].id()
        cluster_val = self.features[idx][self.dialog.cluster_field]
        self.clustered_layer.startEditing()
        self.clustered_layer.changeAttributeValue(fid, self.cluster_idx, cluster_val)
        self.clustered_layer.commitChanges()
        self.clustered_layer.triggerRepaint()

        self.rubber_alpha = self.dialog.initial_alpha
        self.alpha_timer.start(int(self.timer.interval() / 10))
        self.i += 1

    def grow_circle(self):
        if self.i >= len(self.visit_order):
            self.alpha_timer.stop()
            self.rubber.reset(QgsWkbTypes.PolygonGeometry)
            self.iface.mapCanvas().refresh()
            return

        if self.rubber_alpha >= 255:
            self.alpha_timer.stop()
            self.rubber.reset(QgsWkbTypes.PolygonGeometry)
            self.iface.mapCanvas().refresh()
            return

        self.rubber.reset(QgsWkbTypes.PolygonGeometry)
        pt_canvas = self.transform.transform(self.points[self.visit_order[self.i - 1]])
        progress = self.rubber_alpha / 255.0
        radius = self.dialog.eps * (1 - math.exp(-5 * progress))

        for angle in range(0, 360, 10):
            rad = math.radians(angle)
            x = pt_canvas.x() + radius * math.cos(rad)
            y = pt_canvas.y() + radius * math.sin(rad)
            self.rubber.addPoint(QgsPointXY(x, y))
        self.rubber.closePoints()

        cluster_id = self.features[self.visit_order[self.i - 1]][self.dialog.cluster_field]
        color = self.color_map.get(cluster_id, QColor(0, 0, 0))
        color.setAlpha(self.rubber_alpha)
        self.rubber.setColor(color)
        self.rubber.setFillColor(Qt.transparent)

        self.iface.mapCanvas().refresh()
        self.rubber_alpha += self.dialog.alpha_step

    def pause(self):
        self.timer.stop()
        self.alpha_timer.stop()

    def resume(self):
        if self.i < len(self.visit_order):
            self.timer.start()
            self.rubber_alpha = self.dialog.initial_alpha
            self.alpha_timer.start(int(self.timer.interval() / 10))

    def stop(self):
        self.timer.stop()
        self.alpha_timer.stop()
        self.rubber.reset(QgsWkbTypes.PolygonGeometry)
        for r in self.previous_rubbers:
            r.reset(QgsWkbTypes.PolygonGeometry)
        self.previous_rubbers.clear()
        self.iface.mapCanvas().refresh()
        self.i = 0
        # UI 복구
        self.dialog.minpts_spin.setEnabled(True)
        self.dialog.eps_slider.setEnabled(True)


