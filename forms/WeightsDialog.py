# -*- coding: utf-8 -*-
"""
Dialog for configuring spatial weights.
"""

import os
from qgis.PyQt import uic
from qgis.PyQt.QtWidgets import QDialog

pluginPath = os.path.dirname(__file__)
WIDGET, BASE = uic.loadUiType(os.path.join(pluginPath, 'WeightsDialog.ui'))


class WeightsDialog(BASE, WIDGET):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
