# -*- coding: utf-8 -*-
"""
/***************************************************************************
 UrbanAnalyzer
                                 A QGIS plugin
 This plugin provides data and tools for analyzing urban space
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2018-05-05
        git sha              : $Format:%H$
        copyright            : (C) 2018 by D.J Paek
        email                : 1002jeen@daum.net
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
from urban_analysis.UrbanProvider import UrbanProvider
from qgis.core import QgsApplication

class UrbanAnalyzer:
    def __init__(self, iface):
        self.provider = UrbanProvider()

    def initGui(self):
        QgsApplication.processingRegistry().addProvider(self.provider)

    def unload(self):
        QgsApplication.processingRegistry().removeProvider(self.provider)