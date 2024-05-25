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
from PyQt5.QtCore import pyqtSignal
import os
from qgis.PyQt import uic
from qgis.core import QgsProject, QgsProcessingUtils, QgsMessageLog

pluginPath = os.path.dirname(__file__)
WIDGET, BASE = uic.loadUiType(
    os.path.join(pluginPath, 'VariableWidget.ui'))


class VariableWidget(BASE, WIDGET):

    hasChanged = pyqtSignal()

    def __init__(self):
        super(VariableWidget, self).__init__(None)
        self.setupUi(self)
        self.attrs = []
        self.v_type = 'geom'
        self.normalized = False
        self.buttonGroup.buttonClicked.connect(self.setVariables)
        self.fieldCB.checkedItemsChanged.connect(self.changeAttributes)
        self.checkBox.stateChanged.connect(self.is_normalized)

    def setSource(self, source):
        if not source:
            return
        self.fieldCB.clear()
        self.source = source
        numeric_fields = [f.name() for f in source.fields() if f.isNumeric()]
        self.fieldCB.addItems(numeric_fields)
        self.geom.click()

    def setVariables(self):
        if self.geom.isChecked():
            self.v_type = 'geom'
            self.attrs = []
            self.fieldCB.setEnabled(False)
        else:
            self.v_type = 'attrs'
            self.attrs = self.fieldCB.checkedItems()
            self.fieldCB.setEnabled(True)
        self.hasChanged.emit()

    def changeAttributes(self):
        self.attrs = self.fieldCB.checkedItems()
        self.hasChanged.emit()

    def is_normalized(self):
        self.normalized = True if self.checkBox.isChecked() else False
        self.hasChanged.emit()

    def setValue(self, value):
        return True

    def value(self):
        v = [self.v_type, self.attrs, self.normalized]
        return v


class VariableWidgetWrapper(WidgetWrapper):

    def __init__(self, param, dialog, row=3, col=0, **kwargs):
        super().__init__(param, dialog, row, col, **kwargs)
        self.context = dataobjects.createContext()

    def createWidget(self):
        if self.dialogType == DIALOG_STANDARD:
            widget = VariableWidget()
            widget.hasChanged.connect(lambda: self.widgetValueHasChanged.emit(self))
            return widget

    def postInitialize(self, wrappers):
        if self.dialogType != DIALOG_STANDARD:
            return
        for wrapper in wrappers:
            if wrapper.parameterDefinition().name() == self.param.layer_param:
                self.setSource(wrapper.parameterValue())
                wrapper.widgetValueHasChanged.connect(self.layerChanged)

    def layerChanged(self, wrapper):
        self.setSource(wrapper.parameterValue())

    def setSource(self, source):
        source = QgsProcessingUtils.variantToSource(source, self.context)
        self.widget.setSource(source)

    def setValue(self, value):
        self.widget.setValue(value)
        
    def value(self):
        return self.widget.value()

