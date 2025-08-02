# -*- coding: utf-8 -*-
"""Dialog for configuring spatial weights."""

import os
from qgis.PyQt import uic
from qgis.PyQt.QtWidgets import QDialog
from qgis.PyQt.QtCore import QVariant

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
            self.xCoordCombo.addItems(field_names)
            self.yCoordCombo.addItems(field_names)

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
                w = weights.util.block_weights(gdf[block_col], ids=gdf[id_field])
            order = self.orderSpin.value()
            if order > 1:
                w = w.higher_order(order, self.includeLowerCheck.isChecked())
            w.transform = 'R'
            return w

        if self.distanceTypeTabs.currentIndex() == 0:
            coords = list(zip(gdf[self.xCoordCombo.currentText()],
                              gdf[self.yCoordCombo.currentText()]))
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

        raise NotImplementedError('Variable based distance is not implemented')