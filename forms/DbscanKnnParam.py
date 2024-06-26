# -*- coding: utf-8 -*-
"""
/***************************************************************************
SpatialAnalyzer
                                 A QGIS plugin
 This plugin provides data and tools for analyzing space
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

from qgis.core import QgsProcessingParameterMatrix

class ParameterKnn(QgsProcessingParameterMatrix):
    def __init__(self, name='', description='', layer_param=None, k_param=None, default=None, optional=False):
        QgsProcessingParameterMatrix.__init__(self, name, description)
        self.layer_param = layer_param
        self.k_param = k_param

    def clone(self):
        copy = ParameterKnn(self.name(), self.description(), self.layer_param, self.k_param)
        return copy
