from qgis.core import (
    QgsProject,
    QgsWkbTypes,
    QgsMarkerSymbol,
    QgsRendererCategory,
    QgsCategorizedSymbolRenderer,
    QgsCoordinateTransform,
    QgsPointXY,
)
from qgis.gui import QgsRubberBand
from qgis.utils import iface
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor

import math
import colorsys
from collections import defaultdict


class DBSCANAnimator:
    """Animate DBSCAN clustering results on the map canvas."""

    active_instance = None
    
    def __init__(self, dialog):
        self.dialog = dialog
        self.layer = dialog.layer
        self.timer = QTimer()
        self.timer.setInterval(dialog.timer_interval)
        self.timer.timeout.connect(self.step)

        self.index = 0
        self.features = []
        self.color_map = {}
        self.transform = None
        self.previous_rubbers = []
        self.current_rb = None
        self.current_pt = None
        self.progress = 0.0
        DBSCANAnimator.active_instance = self

    # ------------------------------------------------------------------
    @classmethod
    def clear_active(cls):
        if cls.active_instance:
            cls.active_instance.clear()
            cls.active_instance = None


    # ------------------------------------------------------------------
    def generate_color(self, index, total):
        h = (index / max(1, total)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        return QColor(int(r * 255), int(g * 255), int(b * 255))

    # ------------------------------------------------------------------
    def run_dbscan_and_prepare(self):
        """Run DBSCAN and prepare animation objects."""
        self.clear_rubbers()
        self.index = 0
        cluster_layer = self.dialog.build_cluster_layer()

        cluster_field = self.dialog.cluster_field
        feats = list(cluster_layer.getFeatures())
        clusters = defaultdict(list)
        for f in feats:
            clusters[f[cluster_field]].append(f)

        def order_cluster(features):
            remaining = features[:]
            ordered = []
            if not remaining:
                return ordered
            current = remaining.pop(0)
            ordered.append(current)
            while remaining:
                last_pt = ordered[-1].geometry().asPoint()
                idx = min(
                    range(len(remaining)),
                    key=lambda i: math.hypot(
                        last_pt.x() - remaining[i].geometry().asPoint().x(),
                        last_pt.y() - remaining[i].geometry().asPoint().y(),
                    ),
                )
                ordered.append(remaining.pop(idx))
            return ordered

        ordered_features = []
        for cid in [c for c in clusters.keys() if c != -1]:
            ordered_features.extend(order_cluster(clusters[cid]))
        if -1 in clusters:
            ordered_features.extend(order_cluster(clusters[-1]))

        self.features = ordered_features

        cluster_ids = sorted(
            {f[cluster_field] for f in self.features if f[cluster_field] not in (None, -1)}
        )
        self.color_map = {
            cid: self.generate_color(i, len(cluster_ids))
            for i, cid in enumerate(cluster_ids)
        }

        categories = [
            QgsRendererCategory(
                cid,
                QgsMarkerSymbol.createSimple(
                    {"name": "circle", "color": color.name(), "size": self.dialog.marker_size}
                ),
                f"Cluster {cid}",
            )
            for cid, color in self.color_map.items()
        ]
        categories.append(
            QgsRendererCategory(
                -1,
                QgsMarkerSymbol.createSimple(
                    {"name": "circle", "color": "black", "size": self.dialog.marker_size}
                ),
                "Noise",
            )
        )
        cluster_layer.setRenderer(
            QgsCategorizedSymbolRenderer(cluster_field, categories)
        )

        canvas = iface.mapCanvas()
        self.transform = QgsCoordinateTransform(
            self.layer.crs(),
            canvas.mapSettings().destinationCrs(),
            QgsProject.instance(),
        )
        cluster_layer = None
        self.timer.start()

    # ------------------------------------------------------------------
    def step(self):
        if self.index >= len(self.features) and not self.current_rb:
            self.timer.stop()
            self.dialog.animation_finished()
            return

        if not self.current_rb:
            feat = self.features[self.index]
            pt_canvas = self.transform.transform(feat.geometry().asPoint())
            cluster_id = feat[self.dialog.cluster_field]
            color = self.color_map.get(cluster_id, QColor(0, 0, 0))
            color.setAlpha(self.dialog.fill_alpha)

            self.current_pt = pt_canvas
            self.current_rb = QgsRubberBand(iface.mapCanvas(), QgsWkbTypes.PolygonGeometry)
            self.current_rb.setWidth(0)
            self.current_rb.setColor(Qt.transparent)
            self.current_rb.setFillColor(color)
            self.progress = 0.0

        self.progress += 0.3
        if self.progress > 1.0:
            self.progress = 1.0

        radius = self.dialog.eps * (1 - (1 - self.progress) ** 2)
        self.current_rb.reset(QgsWkbTypes.PolygonGeometry)

        for angle in range(0, 360, 10):
            rad = math.radians(angle)
            x = self.current_pt.x() + radius * math.cos(rad)
            y = self.current_pt.y() + radius * math.sin(rad)
            self.current_rb.addPoint(QgsPointXY(x, y))
        self.current_rb.closePoints()

        if self.progress >= 1.0:
            self.previous_rubbers.append(self.current_rb)
            self.current_rb = None
            self.progress = 0.0
            self.index += 1

    # ------------------------------------------------------------------
    def pause(self):
        self.timer.stop()

    # ------------------------------------------------------------------
    def resume(self):
        if self.index < len(self.features):
            self.timer.start()

    # ------------------------------------------------------------------
    def stop(self):
        self.timer.stop()
        self.index = 0
        self.clear_rubbers()
        self.current_rb = None
        self.progress = 0.0
        DBSCANAnimator.active_instance = None

    # ------------------------------------------------------------------
    def clear_rubbers(self):
        if self.current_rb:
            self.current_rb.reset(QgsWkbTypes.PolygonGeometry)
            self.current_rb = None
        for rb in self.previous_rubbers:
            rb.reset(QgsWkbTypes.PolygonGeometry)
        self.previous_rubbers.clear()
        iface.mapCanvas().refresh()

    # ------------------------------------------------------------------
    def clear(self):
        """Clear results from canvas."""
        self.stop()
