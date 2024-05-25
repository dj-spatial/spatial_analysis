# -*- coding: utf-8 -*-
"""
/***************************************************************************
                                 A QGIS plugin
 SpatialAnalyzer
                              -------------------
        git sha              : $Format:%H$
        copyright            : (C) 2017 by D.J Paek
        email                : dj dot paek1 at gmail dot com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

__author__ = 'D.J Paek'
__date__ = 'March 2019'
__copyright__ = '(C) 2019, D.J Paek'

# This will get replaced with a git SHA1 when you do a git archive

__revision__ = '$Format:%H$'

from processing.gui.wrappers import WidgetWrapper, DIALOG_STANDARD
from processing.tools import dataobjects
import os
import tempfile
import numpy as np
from scipy.cluster.vq import kmeans,vq
import plotly
import plotly as plt
import plotly.graph_objs as go
from qgis.PyQt import uic
from qgis.core import Qgis, QgsMessageLog, QgsNetworkAccessManager, QgsProject 
from qgis.gui import QgsMessageBar
from PyQt5.QtCore import QDate
from qgis.PyQt.QtWidgets import QVBoxLayout
from qgis.PyQt.QtWebKit import QWebSettings
from qgis.PyQt.QtWebKitWidgets import QWebView
from qgis.PyQt.QtCore import QUrl, Qt
from PyQt5.QtWidgets import QMessageBox, QTableWidgetItem

pluginPath = os.path.dirname(__file__)
WIDGET, BASE = uic.loadUiType(
    os.path.join(pluginPath, 'GravityAttraction.ui'))


class AttractionWidget(BASE, WIDGET):

    def __init__(self):
        super(AttractionWidget, self).__init__(None)
        self.setupUi(self)

        # Connect signals
        # self.browserBtn.clicked.connect(self.browserVeiw)

    def setFields(self, fields):
        self.attr_factors = fields
        self.tableWidget.clearContents()
        self.tableWidget.setRowCount(len(self.attr_factors))
        for i, f in enumerate(self.attr_factors):
            item = QTableWidgetItem(f)
            item.setTextAlignment(Qt.AlignCenter|Qt.AlignVCenter)
            self.tableWidget.setItem(i, 0, item)
        
    def value(self):
        attr_params = [float(self.tableWidget.item(i, 1).text()) for i in range(len(self.attr_factors))]
        QgsMessageLog.logMessage(str(attr_params))
        return attr_params

class AttractionWidgetWrapper(WidgetWrapper):

    def __init__(self, param, dialog, row=3, col=0, **kwargs):
        super().__init__(param, dialog, row, col, **kwargs)
        self.context = dataobjects.createContext()

    def _panel(self):
        return AttractionWidget()

    def createWidget(self):
        if self.dialogType == DIALOG_STANDARD:
            return self._panel()

    def postInitialize(self, wrappers):
        if self.dialogType != DIALOG_STANDARD:
            return
        for wrapper in wrappers:
            if wrapper.parameterDefinition().name() == self.param.fields_param:
                self.setFields(wrapper.parameterValue())
                wrapper.widgetValueHasChanged.connect(self.fieldsChanged)

    def fieldsChanged(self, wrapper):
        self.setFields(wrapper.parameterValue())

    def setFields(self, fields):
        self.widget.setFields(fields)

    def value(self):
        return self.widget.value()

