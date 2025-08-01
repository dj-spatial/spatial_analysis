# -*- coding: utf-8 -*-
"""
Processing widget wrapper providing a button to open the Weights dialog.
"""

from processing.gui.wrappers import WidgetWrapper, DIALOG_STANDARD
from qgis.PyQt.QtWidgets import QPushButton
from qgis.PyQt.QtCore import QCoreApplication


class WeightsWidgetWrapper(WidgetWrapper):

    def createWidget(self):
        if self.dialogType == DIALOG_STANDARD:
            self.button = QPushButton(self.tr('Weights'))
            self.button.clicked.connect(self.openDialog)
            return self.button

    def openDialog(self):
        from spatial_analysis.forms.WeightsDialog import WeightsDialog
        dlg = WeightsDialog()
        dlg.exec_()

    def value(self):
        return ''

    def setValue(self, value):
        return True

    def tr(self, string):
        return QCoreApplication.translate('WeightsWidgetWrapper', string)
