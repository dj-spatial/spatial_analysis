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
        email                : dj.paek1@gmail.com
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
import os

from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon

from .algs import (
    CentralTendency,
    MeanCenterTracker,
    MedianCenterTracker,
    CentralFeatureTracker,
    StandardDistance,
    StandardDeviationEllipse,
    Gravity,
    Kmeans,
    Hierarchical,
    Dbscan
)

pluginPath = os.path.split(os.path.dirname(__file__))[0]

class SpatialProvider(QgsProcessingProvider):
    def __init__(self):
        super().__init__()
        self.alglist = [
            CentralTendency.CentralTendency(),
            MeanCenterTracker.MeanCenterTracker(),
            MedianCenterTracker.MedianCenterTracker(),
            CentralFeatureTracker.CentralFeatureTracker(),
            StandardDistance.StandardDistance(),
            StandardDeviationEllipse.StandardDeviationEllipse(),
            Gravity.Gravity(),
            Kmeans.Kmeans(),
            Hierarchical.Hierarchical(),
            Dbscan.Dbscan()
        ]
        
    def getAlgs(self):
        return self.alglist

    def id(self, *args, **kwargs):
        return 'spatialAnalyzer'

    def name(self, *args, **kwargs):
        return 'SpatialAnalyzer - Spatial Analysis Toolbox'

    def icon(self):
        return QIcon(os.path.join(pluginPath, 'spatial_analysis', 'icons', 'icon.svg'))

    def svgIconPath(self):
        return os.path.join(pluginPath, 'spatial_analysis', 'icon.svg')

    def loadAlgorithms(self, *args, **kwargs):
        for alg in self.alglist:
            self.addAlgorithm(alg)