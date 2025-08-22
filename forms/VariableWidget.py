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
from qgis.PyQt.QtCore import pyqtSignal, Qt
from qgis.PyQt.QtWidgets import QMenu, QListWidgetItem
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
        self.fieldLW.itemChanged.connect(self.changeAttributes)
        self.checkBox.stateChanged.connect(self.is_normalized)
        self.fieldLW.setContextMenuPolicy(Qt.CustomContextMenu)
        self.fieldLW.customContextMenuRequested.connect(self.showContextMenu)

    def setSource(self, source):
        if not source:
            return
        self.fieldLW.clear()
        self.source = source
        numeric_fields = [f.name() for f in source.fields() if f.isNumeric()]
        for f in numeric_fields:
            item = QListWidgetItem(f)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.fieldLW.addItem(item)
        self.geom.click()

    def setVariables(self):
        if self.geom.isChecked():
            self.v_type = 'geom'
            self.attrs = []
            self.fieldLW.setEnabled(False)
        else:
            self.v_type = 'attrs'
            self.attrs = self.checkedItems()
            self.fieldLW.setEnabled(True)
        self.hasChanged.emit()

    def changeAttributes(self):
        self.attrs = self.checkedItems()
        self.hasChanged.emit()

    def showContextMenu(self, pos):
        menu = QMenu(self)
        menu.addAction(self.tr('Select All'), self.selectAll)
        menu.addAction(self.tr('Clear Selection'), self.clearSelection)
        menu.addAction(self.tr('Toggle Selection'), self.toggleSelection)
        menu.exec_(self.fieldLW.mapToGlobal(pos))

    def selectAll(self):
        for i in range(self.fieldLW.count()):
            item = self.fieldLW.item(i)
            item.setCheckState(Qt.Checked)
        self.changeAttributes()

    def clearSelection(self):
        for i in range(self.fieldLW.count()):
            item = self.fieldLW.item(i)
            item.setCheckState(Qt.Unchecked)
        self.changeAttributes()

    def toggleSelection(self):
        for i in range(self.fieldLW.count()):
            item = self.fieldLW.item(i)
            state = Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked
            item.setCheckState(state)
        self.changeAttributes()

    def is_normalized(self):
        self.normalized = True if self.checkBox.isChecked() else False
        self.hasChanged.emit()

    def setValue(self, value):
        return True

    def value(self):
        v = [self.v_type, self.attrs, self.normalized]
        return v

    def checkedItems(self):
        items = []
        for i in range(self.fieldLW.count()):
            item = self.fieldLW.item(i)
            if item.checkState() == Qt.Checked:
                items.append(item.text())
        return items

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

