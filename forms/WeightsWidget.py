# -*- coding: utf-8 -*-
"""Processing widget wrapper providing a button to open the Weights dialog."""

from processing.gui.wrappers import WidgetWrapper, DIALOG_STANDARD
from processing.tools import dataobjects
from qgis.PyQt.QtWidgets import QPushButton, QWidget, QHBoxLayout
from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import QgsProcessingUtils, QgsMessageLog, Qgis


class WeightsWidgetWrapper(WidgetWrapper):

    def __init__(self, param, dialog, row=0, col=0, **kwargs):
        self.layer_param = kwargs.get('layer_param')
        super().__init__(param, dialog, row, col, **kwargs)
        self.context = dataobjects.createContext()
        self.layer = None
        self.weight_data = None

    def createWidget(self, **_):
        if self.dialogType == DIALOG_STANDARD:
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            self.button = QPushButton(self.tr('Weight Manager'))
            self.button.clicked.connect(self.openDialog)

            layout.addWidget(self.button)
            return container

    def postInitialize(self, wrappers):
        if self.dialogType != DIALOG_STANDARD:
            return
        layer_param = self.layer_param or self.param.metadata().get('widget_wrapper', {}).get('layer_param')
        for w in wrappers:
            if w.parameterDefinition().name() == layer_param:
                self.layer = QgsProcessingUtils.variantToSource(w.parameterValue(), self.context)
                w.widgetValueHasChanged.connect(self.layerChanged)

    def layerChanged(self, wrapper):
        self.layer = QgsProcessingUtils.variantToSource(wrapper.parameterValue(), self.context)

    def openDialog(self):
        if not self.layer:
            return
        from spatial_analysis.forms.WeightsDialog import WeightsDialog
        dlg = WeightsDialog(self.layer)
        if dlg.exec_():
            gdf = self._layer_to_gdf(self.layer)
            try:
                w = dlg.build_weights(gdf)
            except Exception as exc:  # pragma: no cover - dialog runtime
                QgsMessageLog.logMessage(str(exc), 'spatial_analysis', Qgis.Critical)
                return
            summary = dlg.weight_summary(w)
            self.weight_data = {
                'weights': w,
                'id_field': dlg.idFieldCombo.currentText(),
                'summary': summary
            }
            dialog = self.dialog() if callable(self.dialog) else self.dialog
            if dialog and hasattr(dialog, 'loadWeightSummary'):
                dialog.loadWeightSummary(summary)
            self.widgetValueHasChanged.emit(self)

    def _layer_to_gdf(self, layer):
        import geopandas as gpd
        from shapely import wkb

        fields = [f.name() for f in layer.fields()]
        records = []
        for feat in layer.getFeatures():
            attrs = {name: feat[name] for name in fields}
            attrs['geometry'] = wkb.loads(bytes(feat.geometry().asWkb()))
            records.append(attrs)
        return gpd.GeoDataFrame(records, geometry='geometry', crs=layer.sourceCrs().toWkt())

    def value(self):
        return self.weight_data

    def setValue(self, value):
        self.weight_data = value
        dialog = self.dialog() if callable(self.dialog) else self.dialog
        if value and dialog and hasattr(dialog, 'loadWeightSummary'):
            dialog.loadWeightSummary(value.get('summary', ''))
        return True

    def _summarize_weights(self, w):
        import numpy as np
        card = np.array([len(nbrs) for nbrs in w.neighbors.values()])
        return self.tr('n={0}, mean={1:.2f}, min={2}, max={3}').format(
            w.n, card.mean(), card.min(), card.max())

    def tr(self, string):
        return QCoreApplication.translate('WeightsWidgetWrapper', string)
