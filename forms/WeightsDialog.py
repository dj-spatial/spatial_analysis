# -*- coding: utf-8 -*-
"""Dialog for configuring spatial weights."""

import os
from qgis.PyQt import uic
from qgis.PyQt.QtWidgets import QDialog
from qgis.PyQt.QtCore import QVariant
import numpy as np

pluginPath = os.path.dirname(__file__)
WIDGET, BASE = uic.loadUiType(os.path.join(pluginPath, 'WeightsDialog.ui'))


class WeightsDialog(BASE, WIDGET):
    """Dialog to configure spatial weights using PySAL."""

    def __init__(self, layer=None, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        self.layer = layer
        
        if layer is not None:
            field_names = [f.name() for f in layer.fields()]
            self.idFieldCombo.addItems(field_names)
            numeric_types = {QVariant.Int, QVariant.UInt, QVariant.LongLong,
                             QVariant.ULongLong, QVariant.Double}
            numeric_fields = [f.name() for f in layer.fields()
                              if f.type() in numeric_types]
            self.variablesList.addItems(numeric_fields)
            id_types = {QVariant.Int, QVariant.UInt, QVariant.LongLong,
                        QVariant.ULongLong, QVariant.String}
            block_fields = [f.name() for f in layer.fields()
                            if f.type() in id_types]
            self.blockColumnCombo.addItems(block_fields)

            # compute default bandwidth from centroid distances
            coords = []
            for feat in layer.getFeatures():
                c = feat.geometry().centroid().asPoint()
                coords.append((c.x(), c.y()))
            bw = self._calc_default_bandwidth(coords)
            self.bandwidthSpin.setValue(bw)
            self.kernelBandwidthSpin.setValue(bw)

        self.contiguityCombo.currentTextChanged.connect(self._toggle_block)
        self.orderSpin.valueChanged.connect(self._toggle_include_lower)
        self.bandInverseCheck.toggled.connect(self.bandPowerSpin.setEnabled)
        self.knnInverseCheck.toggled.connect(self.knnPowerSpin.setEnabled)
        self.bandwidthSlider.valueChanged.connect(lambda v: self.bandwidthSpin.setValue(float(v)))
        self.bandwidthSpin.valueChanged.connect(lambda v: self.bandwidthSlider.setValue(int(v)))
        self.kernelBandwidthSlider.valueChanged.connect(lambda v: self.kernelBandwidthSpin.setValue(float(v)))
        self.kernelBandwidthSpin.valueChanged.connect(lambda v: self.kernelBandwidthSlider.setValue(int(v)))
        self.variablesList.itemSelectionChanged.connect(self._update_var_bandwidth)
        self.transformCombo.currentIndexChanged.connect(lambda _: self._update_var_bandwidth())
        self.varMetricCombo.currentIndexChanged.connect(lambda _: self._update_var_bandwidth())
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        # defaults per PySAL recommendations
        self.knnSpin.setValue(4)
        self.kernelNeighborsSpin.setValue(15)
        self.adaptiveBandwidthRadio.setChecked(True)
        self.bandwidthValueRadio.setChecked(False)

        # initialise widget states
        self._toggle_block(self.contiguityCombo.currentText())
        self._toggle_include_lower(self.orderSpin.value())
        self.bandwidthSlider.setRange(int(self.bandwidthSpin.minimum()), int(self.bandwidthSpin.maximum()))
        self.kernelBandwidthSlider.setRange(int(self.kernelBandwidthSpin.minimum()), int(self.kernelBandwidthSpin.maximum()))
        self.bandwidthSlider.setValue(int(self.bandwidthSpin.value()))
        self.kernelBandwidthSlider.setValue(int(self.kernelBandwidthSpin.value()))
        self.rowStdCheck.setChecked(True)


    def _toggle_block(self, text):
        enabled = text == 'Block'
        self.blockLabel.setEnabled(enabled)
        self.blockColumnCombo.setEnabled(enabled)

    def _toggle_include_lower(self, value):
        enabled = value >= 2
        self.includeLowerCheck.setEnabled(enabled)
        if not enabled:
            self.includeLowerCheck.setChecked(False)

    def _calc_default_bandwidth(self, coords, p=2):
        """Return max of each point's nearest neighbour distance."""
        coords = np.asarray(coords, dtype=float)
        if len(coords) <= 1:
            return 0.0
        if p == 2:
            diff = coords[:, None, :] - coords[None, :, :]
            dist = np.sqrt((diff ** 2).sum(-1))
        else:
            diff = np.abs(coords[:, None, :] - coords[None, :, :])
            dist = diff.sum(-1)
        np.fill_diagonal(dist, np.inf)
        return float(dist.min(axis=1).max())

    def _update_var_bandwidth(self):
        if self.layer is None:
            return
        vars_selected = [i.text() for i in self.variablesList.selectedItems()]
        if not vars_selected:
            return
        data = []
        for feat in self.layer.getFeatures():
            row = []
            for v in vars_selected:
                val = feat[v]
                if val is None:
                    break
                row.append(float(val))
            else:
                data.append(row)
        if len(data) < 2:
            return
        data = np.asarray(data, dtype=float)
        transform = self.transformCombo.currentText()
        if transform == 'Standardize (Z)':
            data = (data - data.mean(axis=0)) / data.std(axis=0)
        elif transform == 'Standardize (MAD)':
            med = np.median(data, axis=0)
            mad = np.median(np.abs(data - med), axis=0)
            mad[mad == 0] = 1
            data = (data - med) / mad
        elif transform == 'Demean':
            data = data - data.mean(axis=0)
        metric = self.varMetricCombo.currentText()
        p = 2 if metric == 'Euclidean' else 1
        bw = self._calc_default_bandwidth(data, p=p)
        self.bandwidthSpin.setValue(bw)
        self.bandwidthSlider.setValue(int(bw))
        self.kernelBandwidthSpin.setValue(bw)
        self.kernelBandwidthSlider.setValue(int(bw))

    def build_weights(self, gdf):
        """Return a PySAL weights object based on dialog settings."""
        try:
            from libpysal import weights
        except ImportError as exc:
            raise ImportError('PySAL is required to build weights') from exc

        id_field = self.idFieldCombo.currentText()
        ids = gdf[id_field].tolist()

        if self.mainTab.currentIndex() == 0:
            method = self.contiguityCombo.currentText()
            if method == 'Queen':
                w = weights.Queen.from_dataframe(gdf, ids=gdf[id_field])
            elif method == 'Rook':
                w = weights.Rook.from_dataframe(gdf, ids=gdf[id_field])
            else:
                block_col = self.blockColumnCombo.currentText()
                gdf[block_col] = gdf[block_col].astype("category")
                w = weights.util.block_weights(gdf[block_col], ids=gdf[id_field])
            order = self.orderSpin.value()
            if order > 1:
                w = w.higher_order(order, self.includeLowerCheck.isChecked())
            if self.rowStdCheck.isChecked():
                w.transform = 'R'
            else:
                w.transform = 'B'
            return w

        method_idx = self.methodTabs.currentIndex()
        if self.distanceTypeTabs.currentIndex() == 0:
            coords = list(zip(gdf.geometry.centroid.x,
                              gdf.geometry.centroid.y))
            if method_idx == 0:
                threshold = self.bandwidthSpin.value()
                if self.bandInverseCheck.isChecked():
                    w = weights.DistanceBand(coords, threshold=threshold,
                                             alpha=self.bandPowerSpin.value(),
                                             binary=False, ids=ids)
                else:
                    w = weights.DistanceBand(coords, threshold=threshold,
                                             binary=True, ids=ids)
            elif method_idx == 1:
                k = self.knnSpin.value()
                w = weights.KNN(coords, k=k, ids=ids)
            else:
                function = self.kernelFunctionCombo.currentText().lower()
                bw = None
                k = None
                fixed = True
                if self.adaptiveBandwidthRadio.isChecked():
                    k = self.kernelNeighborsSpin.value()
                    fixed = False
                elif self.maxKnnRadio.isChecked():
                    k = self.kernelNeighborsSpin.value()
                    bw = 'max'
                    fixed = False
                else:
                    bw = self.kernelBandwidthSpin.value()
                params = dict(data=coords, function=function, ids=ids, fixed=fixed)
                if bw is not None:
                    params['bandwidth'] = bw
                if k is not None:
                    params['k'] = k
                w = weights.Kernel(**params)
        else:
            # variable-based distance
            vars_selected = [i.text() for i in self.variablesList.selectedItems()]
            if not vars_selected:
                raise ValueError('Select at least one variable')
            data = gdf[vars_selected].values.astype(float)
            transform = self.transformCombo.currentText()
            if transform == 'Standardize (Z)':
                data = (data - data.mean(axis=0)) / data.std(axis=0)
            elif transform == 'Standardize (MAD)':
                med = np.median(data, axis=0)
                mad = np.median(np.abs(data - med), axis=0)
                mad[mad == 0] = 1
                data = (data - med) / mad
            elif transform == 'Demean':
                data = data - data.mean(axis=0)
            coords = data.tolist()
            metric = self.varMetricCombo.currentText()
            p = 2 if metric == 'Euclidean' else 1
            if method_idx == 0:
                threshold = self.bandwidthSpin.value()
                if self.bandInverseCheck.isChecked():
                    w = weights.DistanceBand(coords, threshold=threshold, p=p,
                                             alpha=self.bandPowerSpin.value(),
                                             binary=False, ids=ids)
                else:
                    w = weights.DistanceBand(coords, threshold=threshold, p=p, binary=True, ids=ids)
            elif method_idx == 1:
                k = self.knnSpin.value()
                w = weights.KNN(coords, k=k, p=p, ids=ids)
            else:
                function = self.kernelFunctionCombo.currentText().lower()
                bw = None
                k = None
                fixed = True
                if self.adaptiveBandwidthRadio.isChecked():
                    k = self.kernelNeighborsSpin.value()
                    fixed = False
                elif self.maxKnnRadio.isChecked():
                    k = self.kernelNeighborsSpin.value()
                    bw = 'max'
                    fixed = False
                else:
                    bw = self.kernelBandwidthSpin.value()
                params = dict(data=coords, function=function, ids=ids, p=p, fixed=fixed)
                if bw is not None:
                    params['bandwidth'] = bw
                if k is not None:
                    params['k'] = k
                w = weights.Kernel(**params)

        if self.rowStdCheck.isChecked():
            w.transform = 'R'
        else:
            w.transform = 'B'
        return w


    def weight_summary(self, w):
        """Return a text summary describing the selected weights."""
        lines = []
        if self.mainTab.currentIndex() == 0:
            lines.append('Contiguity Based')
            lines.append(f"Method: {self.contiguityCombo.currentText()}")
            lines.append(f"Precision threshold: {self.precisionSpin.value()}")
            order = self.orderSpin.value()
            lines.append(f"Order: {order}")
            if order > 1:
                inc = 'Yes' if self.includeLowerCheck.isChecked() else 'No'
                lines.append(f"Include lower orders: {inc}")
        else:
            lines.append('Distance Based')
            if self.distanceTypeTabs.currentIndex() == 0:
                lines.append('Type: Geometric Centroid')
                lines.append(f"Distance metric: {self.geomMetricCombo.currentText()}")
            else:
                lines.append('Type: Variables')
                vars_selected = [i.text() for i in self.variablesList.selectedItems()]
                lines.append('Variables: {}'.format(', '.join(vars_selected)))
                lines.append(f"Transform: {self.transformCombo.currentText()}")
                lines.append(f"Distance metric: {self.varMetricCombo.currentText()}")

            m_idx = self.methodTabs.currentIndex()
            if m_idx == 0:
                lines.append('Method: Distance Band')
                lines.append(f"Bandwidth: {self.bandwidthSpin.value()}")
                if self.bandInverseCheck.isChecked():
                    lines.append(f"Inverse distance power: {self.bandPowerSpin.value()}")
            elif m_idx == 1:
                lines.append('Method: K-nearest neighbors')
                lines.append(f"k: {self.knnSpin.value()}")
                if self.knnInverseCheck.isChecked():
                    lines.append(f"Inverse distance power: {self.knnPowerSpin.value()}")
            else:
                lines.append('Method: Kernel')
                lines.append(f"Function: {self.kernelFunctionCombo.currentText()}")
                if self.adaptiveBandwidthRadio.isChecked():
                    lines.append(f"Adaptive bandwidth k: {self.kernelNeighborsSpin.value()}")
                elif self.maxKnnRadio.isChecked():
                    lines.append(f"Max knn bandwidth k: {self.kernelNeighborsSpin.value()}")
                else:
                    lines.append(f"Bandwidth: {self.kernelBandwidthSpin.value()}")
                lines.append(f"Diagonal treatment: {self.diagOptionCombo.currentText()}")

        lines.extend([
            f"Observations: {w.n}",
            f"Min neighbors: {w.min_neighbors}",
            f"Max neighbors: {w.max_neighbors}",
            f"Mean neighbors: {w.mean_neighbors:.2f}",
            f"Percent nonzero: {w.pct_nonzero:.2f}",
            f"Islands: {len(w.islands)}",
        ])
        return '\n'.join(lines)