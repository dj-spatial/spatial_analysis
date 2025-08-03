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

        if layer is not None:
            field_names = [f.name() for f in layer.fields()]
            self.idFieldCombo.addItems(field_names)
            self.variablesList.addItems(field_names)
            id_types = {QVariant.Int, QVariant.UInt, QVariant.LongLong,
                        QVariant.ULongLong, QVariant.String}
            block_fields = [f.name() for f in layer.fields()
                            if f.type() in id_types]
            self.blockColumnCombo.addItems(block_fields)


        self.contiguityCombo.currentTextChanged.connect(self._toggle_block)
        self.orderSpin.valueChanged.connect(self._toggle_include_lower)
        self.bandInverseCheck.toggled.connect(self.bandPowerSpin.setEnabled)
        self.knnInverseCheck.toggled.connect(self.knnPowerSpin.setEnabled)
        self.bandwidthSlider.valueChanged.connect(lambda v: self.bandwidthSpin.setValue(float(v)))
        self.bandwidthSpin.valueChanged.connect(lambda v: self.bandwidthSlider.setValue(int(v)))
        self.kernelBandwidthSlider.valueChanged.connect(lambda v: self.kernelBandwidthSpin.setValue(float(v)))
        self.kernelBandwidthSpin.valueChanged.connect(lambda v: self.kernelBandwidthSlider.setValue(int(v)))
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        # initialise widget states
        self._toggle_block(self.contiguityCombo.currentText())
        self._toggle_include_lower(self.orderSpin.value())
        self.bandwidthSlider.setRange(int(self.bandwidthSpin.minimum()), int(self.bandwidthSpin.maximum()))
        self.kernelBandwidthSlider.setRange(int(self.kernelBandwidthSpin.minimum()), int(self.kernelBandwidthSpin.maximum()))
        self.bandwidthSlider.setValue(int(self.bandwidthSpin.value()))
        self.kernelBandwidthSlider.setValue(int(self.kernelBandwidthSpin.value()))

    def _toggle_block(self, text):
        enabled = text == 'Block'
        self.blockLabel.setEnabled(enabled)
        self.blockColumnCombo.setEnabled(enabled)

    def _toggle_include_lower(self, value):
        enabled = value >= 2
        self.includeLowerCheck.setEnabled(enabled)
        if not enabled:
            self.includeLowerCheck.setChecked(False)

    def build_weights(self, gdf):
        """Return a PySAL weights object based on dialog settings."""
        try:
            from libpysal import weights
        except ImportError as exc:
            raise ImportError('PySAL is required to build weights') from exc

        id_field = self.idFieldCombo.currentText()

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
            w.transform = 'R'
            return w

        method_idx = self.methodTabs.currentIndex()
        if self.distanceTypeTabs.currentIndex() == 0:
            coords = list(zip(gdf.geometry.centroid.x,
                              gdf.geometry.centroid.y))
            method_idx = self.geomMethodTabs.currentIndex()
            if method_idx == 0:
                threshold = self.bandwidthSpin.value()
                if self.bandInverseCheck.isChecked():
                    w = weights.DistanceBand(coords, threshold=threshold,
                                             alpha=self.bandPowerSpin.value(),
                                             binary=False)
                else:
                    w = weights.DistanceBand(coords, threshold=threshold,
                                             binary=True)
            elif method_idx == 1:
                k = self.knnSpin.value()
                w = weights.KNN(coords, k=k)
            else:
                function = self.kernelFunctionCombo.currentText().lower()
                bw = None
                k = None
                if self.adaptiveBandwidthRadio.isChecked():
                    k = self.kernelNeighborsSpin.value()
                elif self.maxKnnRadio.isChecked():
                    k = self.kernelNeighborsSpin.value()
                    bw = 'max'
                else:
                    bw = self.kernelBandwidthSpin.value()
                w = weights.Kernel(coords, function=function, bandwidth=bw, k=k)
            return w

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
                                         binary=False)
            else:
                w = weights.DistanceBand(coords, threshold=threshold, p=p, binary=True)
        elif method_idx == 1:
            k = self.knnSpin.value()
            w = weights.KNN(coords, k=k, p=p)
        else:
            function = self.kernelFunctionCombo.currentText().lower()
            bw = None
            k = None
            if self.adaptiveBandwidthRadio.isChecked():
                k = self.kernelNeighborsSpin.value()
            elif self.maxKnnRadio.isChecked():
                k = self.kernelNeighborsSpin.value()
                bw = 'max'
            else:
                bw = self.kernelBandwidthSpin.value()
            w = weights.Kernel(coords, function=function, bandwidth=bw, k=k, p=p)
        return w
