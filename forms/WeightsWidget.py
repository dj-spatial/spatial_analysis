# -*- coding: utf-8 -*-
"""Processing widget wrapper providing a button to open the Weights dialog."""

from processing.gui.wrappers import WidgetWrapper, DIALOG_STANDARD
from processing.tools import dataobjects
from qgis.PyQt.QtWidgets import (
    QPushButton,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QSizePolicy
)
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
            layout = QVBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            self.testGroup = QGroupBox(self.tr(''))
            self.testGroup.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

            group_layout = QHBoxLayout()
            group_layout.setContentsMargins(4, 4, 4, 4)
            group_layout.setSpacing(4)   # 버튼 사이 여백만 남김

            self.button = QPushButton(self.tr('Weight Manager'))
            self.button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

            self.button.clicked.connect(self.openDialog)
            group_layout.addWidget(self.button)

            self.testButton = QPushButton(self.tr('Test'))
            self.testButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.testButton.clicked.connect(self.openTestDialog)
            group_layout.addWidget(self.testButton)

            # group_layout.addStretch()
            self.testGroup.setLayout(group_layout)
            layout.addWidget(self.testGroup)

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
            data = dlg.weight_data
            if data:
                self.weight_data = data
                self.widgetValueHasChanged.emit(self)

    def openTestDialog(self):
        dlg_parent = self.dialog() if callable(self.dialog) else self.dialog
        if not self.weight_data:
            if dlg_parent:
                dlg_parent.messageBar().pushInfo(
                    '',
                    self.tr('Weight Manager를 실행해 Weight Matrix를 만드세요')
                )
            return
        from spatial_analysis.forms.SpatialAutocorrelationDialog import SpatialAutocorrelationDialog
        dlg = SpatialAutocorrelationDialog(self.layer, self.weight_data, dlg_parent)
        dlg.exec_()

    def value(self):
        return self.weight_data

    def setValue(self, value):
        self.weight_data = value
        dialog = self.dialog() if callable(self.dialog) else self.dialog
        return True

    def tr(self, string):
        return QCoreApplication.translate('WeightsWidgetWrapper', string)
