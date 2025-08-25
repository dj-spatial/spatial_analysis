from qgis.core import (
    QgsCategorizedSymbolRenderer,
    QgsMarkerSymbol,
    QgsProject,
    QgsRendererCategory,
    QgsVectorLayer,
    QgsWkbTypes,
    QgsMapLayerProxyModel,
    QgsRenderContext,
    QgsField,
    QgsFeature,
)
from qgis import processing
from qgis.utils import iface
from qgis.gui import QgsMapLayerComboBox
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QSlider,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QSpinBox,
    QMessageBox,
    QFrame,
)
from PyQt5.QtCore import Qt, QVariant
from PyQt5.QtGui import QPixmap, QColor

import math
import matplotlib.pyplot as plt
import tempfile
import colorsys

from .dbscan_animator import DBSCANAnimator
from .VariableWidget import VariableWidget


class ParameterControlDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        DBSCANAnimator.clear_active()
        self.iface = iface
        iface.messageBar().clearWidgets()

        self.setWindowTitle("DBSCAN Parameters")
        self.setMinimumWidth(400)

        self.animator = None
        self.layer = None
        self.layer_id = None

        self.min_points = 4
        self.cluster_field = "CLUSTER_ID"
        self.eps = None
        self.eps_min = 1.0
        self.eps_max = 10.0
        self.eps_factor = 10
        self.timer_interval = 50
        self.fill_alpha = 50
        self.marker_size = "0.5"
        self.sorted_knn_dists = []

        layout = QVBoxLayout()
        self.control_widgets = []

        # 레이어 콤보박스
        self.layer_combo = QgsMapLayerComboBox()
        self.layer_combo.setFilters(QgsMapLayerProxyModel.VectorLayer)
        self.layer_combo.currentIndexChanged.connect(
            lambda _=0: self.on_layer_changed(self.layer_combo.currentLayer())
        )
        layout.addWidget(self.layer_combo)

        # Variable selection (geometry or attributes + normalization)
        self.variable_widget = VariableWidget()
        self.variable_widget.hasChanged.connect(self.on_variable_changed)
        layout.addWidget(self.variable_widget)
        self.control_widgets.append(self.variable_widget)

        # MinPts
        self.minpts_label = QLabel("MinPts: -")
        layout.addWidget(self.minpts_label)
        self.control_widgets.append(self.minpts_label)

        self.minpts_spin = QSpinBox()
        self.minpts_spin.setMinimum(1)
        self.minpts_spin.setMaximum(100)
        self.minpts_spin.valueChanged.connect(self.update_minpts)
        layout.addWidget(self.minpts_spin)
        self.control_widgets.append(self.minpts_spin)

        # Epsilon
        self.eps_label = QLabel("Epsilon: Not set")
        layout.addWidget(self.eps_label)
        self.control_widgets.append(self.eps_label)
        
        self.eps_slider = QSlider(Qt.Horizontal)
        self.eps_slider.valueChanged.connect(self.update_eps)
        layout.addWidget(self.eps_slider)
        self.control_widgets.append(self.eps_slider)

        self.eps_ratio_label = QLabel("")
        layout.addWidget(self.eps_ratio_label)
        self.control_widgets.append(self.eps_ratio_label)

        # Elbow Plot
        self.plot_label = QLabel()
        layout.addWidget(self.plot_label)
        self.control_widgets.append(self.plot_label)

        # Animation Control separator
        line_layout = QHBoxLayout()
        line_left = QFrame()
        line_left.setFrameShape(QFrame.HLine)
        line_left.setFrameShadow(QFrame.Sunken)
        line_label = QLabel("Animation Control")
        line_right = QFrame()
        line_right.setFrameShape(QFrame.HLine)
        line_right.setFrameShadow(QFrame.Sunken)
        line_layout.addWidget(line_left)
        line_layout.addWidget(line_label)
        line_layout.addWidget(line_right)
        layout.addLayout(line_layout)

        # Interval
        self.interval_label = QLabel(f"Interval: {self.timer_interval} ms")
        layout.addWidget(self.interval_label)
        self.control_widgets.append(self.interval_label)

        self.interval_slider = QSlider(Qt.Horizontal)
        self.interval_slider.setMinimum(1)
        self.interval_slider.setMaximum(100)
        self.interval_slider.setValue(self.timer_interval)
        self.interval_slider.valueChanged.connect(self.update_speed)
        layout.addWidget(self.interval_slider)
        self.control_widgets.append(self.interval_slider)

        # Fill Opacity
        self.fill_alpha_label = QLabel(f"Fill Opacity: {round(self.fill_alpha / 255 * 100)}%")
        layout.addWidget(self.fill_alpha_label)
        self.control_widgets.append(self.fill_alpha_label)

        self.fill_alpha_slider = QSlider(Qt.Horizontal)
        self.fill_alpha_slider.setMinimum(0)
        self.fill_alpha_slider.setMaximum(255)
        self.fill_alpha_slider.setValue(self.fill_alpha)
        self.fill_alpha_slider.valueChanged.connect(self.update_alpha)
        layout.addWidget(self.fill_alpha_slider)
        self.control_widgets.append(self.fill_alpha_slider)

        # Buttons
        btns = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play")
        self.pause_btn = QPushButton("⏸ Pause")
        self.stop_btn = QPushButton("■ Stop")
        self.clear_btn = QPushButton("Clear")
        self.run_btn = QPushButton("Run")
        self.play_btn.clicked.connect(self.play)
        self.pause_btn.clicked.connect(self.pause)
        self.stop_btn.clicked.connect(self.stop)
        self.clear_btn.clicked.connect(self.clear_canvas)
        self.run_btn.clicked.connect(self.run_dbscan)
        btns.addWidget(self.play_btn)
        btns.addWidget(self.pause_btn)
        btns.addWidget(self.stop_btn)
        btns.addWidget(self.clear_btn)
        btns.addWidget(self.run_btn)
        layout.addLayout(btns)
        self.control_widgets.extend([self.play_btn, self.pause_btn, self.stop_btn, self.clear_btn, self.run_btn])

        self.setLayout(layout)

        QgsProject.instance().layerWillBeRemoved.connect(self.on_project_layer_removed)
        self.toggle_controls(False)
        self.on_layer_changed(self.layer_combo.currentLayer())

    # ------------------------------------------------------------------
    def showEvent(self, event):
        DBSCANAnimator.clear_active()
        super().showEvent(event)

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        DBSCANAnimator.clear_active()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    def toggle_controls(self, enabled):
        for w in self.control_widgets:
            w.setEnabled(enabled)

    # ------------------------------------------------------------------
    def format_eps(self, eps):
        if eps < 1:
            factor = 1_000_000
            trunc = int(eps * factor) / factor
            return f"{trunc:.6f}"
        return f"{eps:.2f}"

    # ------------------------------------------------------------------
    def set_playing(self, playing):
        geom_mode = self.variable_widget.v_type == "geom"
        self.layer_combo.setEnabled(not playing)
        self.run_btn.setEnabled(not playing)
        self.play_btn.setEnabled(geom_mode and not playing)
        self.clear_btn.setEnabled(geom_mode and not playing)
        self.minpts_spin.setEnabled(not playing)
        self.minpts_label.setEnabled(not playing)
        self.eps_slider.setEnabled(not playing)
        self.eps_label.setEnabled(not playing)
        self.interval_slider.setEnabled(geom_mode and not playing)
        self.interval_label.setEnabled(geom_mode and not playing)
        self.fill_alpha_slider.setEnabled(geom_mode and not playing)
        self.fill_alpha_label.setEnabled(geom_mode and not playing)
        self.pause_btn.setEnabled(geom_mode and playing)
        self.stop_btn.setEnabled(geom_mode and playing)

    # ------------------------------------------------------------------
    def on_layer_changed(self, layer):
        self.clear_canvas()
        self.sorted_knn_dists = []
        self.plot_label.clear()
        self.eps_ratio_label.clear()

        if isinstance(layer, QgsVectorLayer):
            self.layer_id = layer.id()
            if layer.geometryType() != QgsWkbTypes.PointGeometry:
                res = processing.run(
                    "native:centroids", {"INPUT": layer, "OUTPUT": "memory:"}
                )
                self.layer = res["OUTPUT"]
            else:
                self.layer = layer

            self.variable_widget.setSource(self.layer)

            # 기본 파라미터 값을 초기화
            self.min_points = 4
            self.timer_interval = 50
            self.fill_alpha = 50

            self.minpts_spin.blockSignals(True)
            self.minpts_spin.setValue(self.min_points)
            self.minpts_spin.blockSignals(False)
            self.minpts_label.setText(f"MinPts: {self.min_points}")

            self.interval_slider.blockSignals(True)
            self.interval_slider.setValue(self.timer_interval)
            self.interval_slider.blockSignals(False)
            self.interval_label.setText(f"Interval: {self.timer_interval} ms")

            self.fill_alpha_slider.blockSignals(True)
            self.fill_alpha_slider.setValue(self.fill_alpha)
            self.fill_alpha_slider.blockSignals(False)
            percent = round(self.fill_alpha / 255 * 100)
            self.fill_alpha_label.setText(f"Fill Opacity: {percent}%")

            renderer = self.layer.renderer()
            sym = None
            if renderer:
                if hasattr(renderer, "symbol"):
                    sym = renderer.symbol()
                elif hasattr(renderer, "symbols"):
                    try:
                        syms = renderer.symbols(QgsRenderContext())
                    except TypeError:
                        syms = renderer.symbols()
                    if syms:
                        sym = syms[0]
            self.marker_size = str(sym.size()) if sym and hasattr(sym, "size") else "2"

            self.eps = self.compute_knn_kth_distance(self.min_points, eps_for_ratio=True)
            self.update_eps_slider()
            self.refresh_plot()
            self.toggle_controls(True)
            self.set_playing(False)
        else:
            self.layer = None
            self.layer_id = None
            self.variable_widget.setSource(None)
            self.toggle_controls(False)
            self.eps_label.setText("Epsilon: Not set")
            self.minpts_label.setText("MinPts: -")
            self.interval_label.setText("Interval: 0 ms")
            self.fill_alpha_label.setText("Fill Opacity: 0%")
            self.plot_label.clear()

    # ------------------------------------------------------------------
    def update_eps_slider(self):
        self.eps_slider.blockSignals(True)
        self.eps_factor = 1_000_000 if self.eps_max < 1 else 10
        self.eps_slider.setMinimum(int(self.eps_min * self.eps_factor))
        self.eps_slider.setMaximum(int(self.eps_max * self.eps_factor))
        self.eps_slider.setValue(int(self.eps * self.eps_factor))
        self.eps_slider.blockSignals(False)
        self.eps_label.setText(f"Epsilon: {self.eps:.2f}")
        self.update_eps_ratio()

    # ------------------------------------------------------------------
    def on_variable_changed(self):
        if not self.layer:
            return
        if self.variable_widget.v_type == "attrs":
            self.clear_canvas()
        self.eps = self.compute_knn_kth_distance(self.min_points)
        self.update_eps_slider()
        self.refresh_plot()
        self.set_playing(False)

    # ------------------------------------------------------------------
    def update_minpts(self, value):
        self.min_points = value
        self.minpts_label.setText(f"MinPts: {value}")
        self.eps = self.compute_knn_kth_distance(self.min_points)
        self.update_eps_slider()
        self.refresh_plot()

    # ------------------------------------------------------------------
    def update_eps(self, value):
        self.eps = value / self.eps_factor
        self.eps_label.setText(f"Epsilon: {self.eps:.2f}")
        self.update_eps_ratio()
        self.refresh_plot()

    # ------------------------------------------------------------------
    def update_eps_ratio(self):
        if self.sorted_knn_dists:
            count = sum(1 for d in self.sorted_knn_dists if d <= self.eps)
            ratio = count / len(self.sorted_knn_dists) * 100
            self.eps_ratio_label.setText(f"{ratio:.0f}% points below epsilon")
        else:
            self.eps_ratio_label.setText("")

    # ------------------------------------------------------------------
    def refresh_plot(self):
        self.plot_label.setPixmap(self.get_elbow_pixmap())

    # ------------------------------------------------------------------
    def update_speed(self, value):
        self.timer_interval = value
        self.interval_label.setText(f"Interval: {value} ms")
        if self.animator:
            self.animator.timer.setInterval(value)

    # ------------------------------------------------------------------
    def update_alpha(self, value):
        self.fill_alpha = value
        percent = round(value / 255 * 100)
        self.fill_alpha_label.setText(f"Fill Opacity: {percent}%")

    # ------------------------------------------------------------------
    def run_dbscan(self):
        if not self.layer:
            QMessageBox.warning(self, "No Layer", "벡터 레이어를 선택해야 합니다.")
            return
        clustered_layer = self.build_cluster_layer()
        QgsProject.instance().addMapLayer(clustered_layer)

    # ------------------------------------------------------------------
    def play(self):
        if not self.layer:
            QMessageBox.warning(self, "No Layer", "벡터 레이어를 선택해야 합니다.")
            return

        if not self.animator or self.animator.layer != self.layer or self.animator.index == 0:
            self.animator = DBSCANAnimator(self)
            self.animator.run_dbscan_and_prepare()
        else:
            self.animator.resume()
        self.set_playing(True)

    # ------------------------------------------------------------------
    def pause(self):
        if self.animator:
            self.animator.pause()
            self.play_btn.setEnabled(True)

    # ------------------------------------------------------------------
    def stop(self):
        if self.animator:
            self.animator.stop()
            self.animator = None
        self.set_playing(False)

    # ------------------------------------------------------------------
    def clear_canvas(self):
        if self.animator:
            self.animator.clear()
            self.animator = None
        self.set_playing(False)

    # ------------------------------------------------------------------
    def animation_finished(self):
        self.set_playing(False)

    # ------------------------------------------------------------------
    def on_project_layer_removed(self, layer_id):
        if self.layer_id and layer_id == self.layer_id:
            self.clear_canvas()
            self.layer = None
            self.layer_id = None
            self.toggle_controls(False)

    # ------------------------------------------------------------------
    def get_feature_matrix(self):
        feats = list(self.layer.getFeatures())
        if self.variable_widget.v_type == "attrs" and self.variable_widget.attrs:
            data = [[f[attr] for attr in self.variable_widget.attrs] for f in feats]
            if self.variable_widget.normalized:
                cols = list(zip(*data))
                means = [sum(col) / len(col) for col in cols]
                stds = [
                    math.sqrt(sum((x - m) ** 2 for x in col) / len(col))
                    for col, m in zip(cols, means)
                ]
                data = [
                    [
                        (val - m) / s if s else 0
                        for val, m, s in zip(row, means, stds)
                    ]
                    for row in data
                ]
            return feats, data
        else:
            points = [f.geometry().asPoint() for f in feats]
            data = [[pt.x(), pt.y()] for pt in points]
            return feats, data

    # ------------------------------------------------------------------
    def _euclidean(self, a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    # ------------------------------------------------------------------
    def _region_query(self, data, idx, eps):
        return [i for i, row in enumerate(data) if self._euclidean(data[idx], row) <= eps]

    # ------------------------------------------------------------------
    def _expand_cluster(self, data, labels, visited, neighbors, cid, eps, min_pts):
        i = 0
        while i < len(neighbors):
            n = neighbors[i]
            if not visited[n]:
                visited[n] = True
                n_neighbors = self._region_query(data, n, eps)
                if len(n_neighbors) >= min_pts:
                    for nb in n_neighbors:
                        if nb not in neighbors:
                            neighbors.append(nb)
            if labels[n] == -1:
                labels[n] = cid
            i += 1

    # ------------------------------------------------------------------
    def _dbscan(self, data, eps, min_pts):
        n = len(data)
        labels = [-1] * n
        visited = [False] * n
        cid = 0
        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = self._region_query(data, i, eps)
            if len(neighbors) < min_pts:
                labels[i] = -1
            else:
                labels[i] = cid
                self._expand_cluster(data, labels, visited, neighbors, cid, eps, min_pts)
                cid += 1
        return labels

    # ------------------------------------------------------------------
    def build_cluster_layer(self):
        feats, data = self.get_feature_matrix()
        labels = self._dbscan(data, self.eps, self.min_points)
        crs = self.layer.crs().authid()
        wkb = QgsWkbTypes.displayString(self.layer.wkbType())
        new_layer = QgsVectorLayer(f"{wkb}?crs={crs}", "DBSCAN_Result", "memory")
        prov = new_layer.dataProvider()
        fields = self.layer.fields()
        fields.append(QgsField(self.cluster_field, QVariant.Int))
        prov.addAttributes(fields)
        new_layer.updateFields()
        new_feats = []
        for f, lab in zip(feats, labels):
            nf = QgsFeature(new_layer.fields())
            nf.setGeometry(f.geometry())
            nf.setAttributes(f.attributes() + [lab])
            new_feats.append(nf)
        prov.addFeatures(new_feats)

        cluster_ids = sorted({lab for lab in labels if lab not in (None, -1)})
        color_map = {}
        for i, cid in enumerate(cluster_ids):
            h = (i / max(1, len(cluster_ids))) % 1.0
            r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
            color_map[cid] = QColor(int(r * 255), int(g * 255), int(b * 255))

        categories = [
            QgsRendererCategory(
                cid,
                QgsMarkerSymbol.createSimple(
                    {"name": "circle", "color": col.name(), "size": self.marker_size}
                ),
                f"Cluster {cid}",
            )
            for cid, col in color_map.items()
        ]
        categories.append(
            QgsRendererCategory(
                -1,
                QgsMarkerSymbol.createSimple(
                    {"name": "circle", "color": "black", "size": self.marker_size}
                ),
                "Noise",
            )
        )
        renderer = QgsCategorizedSymbolRenderer(self.cluster_field, categories)
        new_layer.setRenderer(renderer)
        return new_layer

    # ------------------------------------------------------------------
    def compute_knn_kth_distance(self, k, eps_for_ratio=False):
        feats, data = self.get_feature_matrix()
        kth_dists = []
        for i, row in enumerate(data):
            dist = sorted(
                [
                    self._euclidean(row, data[j])
                    for j in range(len(data))
                    if i != j
                ]
            )
            if len(dist) >= k:
                kth_dists.append(dist[k - 1])
        self.sorted_knn_dists = sorted(kth_dists, reverse=True)
        if kth_dists:
            self.eps_min = min(kth_dists)
            self.eps_max = max(kth_dists)
        return sum(kth_dists) / len(kth_dists) if kth_dists else 10.0

    # ------------------------------------------------------------------
    def get_elbow_pixmap(self):
        fig, ax = plt.subplots(figsize=(4.5, 3.5), facecolor="white")
        ax.set_facecolor("white")
        n = max(len(self.sorted_knn_dists), 1)
        ms = 0.125 * 600 / n
        ms = max(0.1, min(ms, 5.0))
        ax.scatter(
            range(1, n + 1),
            self.sorted_knn_dists,
            s=ms ** 2,
            color="black",
        )
        ax.set_title("k-NN Distance Distribution")
        ax.set_xlabel("Nth Points")
        ax.set_ylabel("k-NN Distance")
        if self.eps is not None:
            ax.axhline(y=self.eps, color="r", linestyle="--", label=f"epsilon = {self.eps:.2f}")
            ax.legend()
        fig.tight_layout()
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig.savefig(tmp.name, facecolor="white")
        plt.close(fig)
        pix = QPixmap(tmp.name)
        return pix
