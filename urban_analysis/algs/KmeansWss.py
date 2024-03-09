# -*- coding: utf-8 -*-
"""
/***************************************************************************
 UrbanReal
                                 A QGIS plugin
 Urban Real Estate Analyzer
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

import os

from qgis.PyQt.QtGui import QIcon

from qgis.core import (QgsProcessing,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterFileDestination)

from processing.algs.qgis.QgisAlgorithm import QgisAlgorithm
import numpy as np
from scipy.cluster.vq import kmeans,vq
import plotly as plt
import plotly.graph_objs as go

pluginPath = os.path.split(os.path.split(os.path.dirname(__file__))[0])[0]


class KmeansWss(QgisAlgorithm):

    INPUT = 'INPUT_POINTS'
    MIN_K = 'MIN_K'
    MAX_K = 'MAX_K'
    OUTPUT = 'OUTPUT'
	
    def icon(self):
        return QIcon(os.path.join(pluginPath, 'urban_analysis', 'icons', 'cluster.png'))

    def group(self):
        return self.tr('Clustering')

    def groupId(self):
        return 'clustering'
    
    def name(self):
        return 'kmeanswss'

    def displayName(self):
        return self.tr('K-Means(WSS Decrease Pattern)')
    
    def msg(self, var):
        return "Type:"+str(type(var))+" repr: "+var.__str__()

    def __init__(self):
        super().__init__()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(self.INPUT,
                                                              self.tr('Points Layer'),
                                                              [QgsProcessing.TypeVectorPoint]))
        self.addParameter(QgsProcessingParameterNumber(self.MAX_K,
                                                       self.tr('Minimum Number of Clusters'),
                                                       QgsProcessingParameterNumber.Integer,
                                                       3, False, 2, 99999999))
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT, self.tr('Scatter plot'), self.tr('HTML files (*.html)')))

    def processAlgorithm(self, parameters, context, feedback):
        feedback.pushInfo(self.tr("Starting Algorithm: '{}'".format(self.displayName())))
        cLayer = self.parameterAsSource(parameters, self.INPUT, context)
        maxK = self.parameterAsInt(parameters, self.MAX_K, context)
        output = self.parameterAsFileOutput(parameters, self.OUTPUT, context)
        
        # get coordinates of point features
        pts=[f.geometry().asPoint() for f in cLayer.getFeatures()]            
        x=[pt[0] for pt in pts]
        y=[pt[1] for pt in pts]
        coords = np.stack([x, y], axis = -1)

        # get wss for each cluster size
        wss = []
        for i in range(1, maxK+1):
            centroids,_ = kmeans(coords, i)  
            cluster, distance = vq(coords, centroids)
            wss.append(sum(distance**2))

        feedback.pushInfo(self.tr("Preparing Graph..."))
        diff = np.diff(wss)
        diff = np.append(0, -diff)
        diff_ratio = diff / wss[0] * 100
        diff_ratio = [("%0.1f"%i+"%") for i in diff_ratio]
        x_axis = [i for i in range(1, maxK+1)]
        trace0 = go.Scatter(x=x_axis, y=wss, name='WSS')
        trace1 = go.Bar(x=x_axis, y=diff, text=diff_ratio, textposition='auto', name='WSS Decrease')
        data = [trace0, trace1]
        layout = go.Layout(title = 'WSS Decrease by the Number of Cluster', xaxis = dict(title='Number of Cluster'))
        fig = go.Figure(data = data, layout = layout)
        plt.offline.plot(fig, filename=output, auto_open=True)
        return {self.OUTPUT: output}