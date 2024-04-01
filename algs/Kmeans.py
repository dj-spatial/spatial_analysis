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

import os

from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtGui import QIcon

from qgis.core import (QgsWkbTypes,
                       QgsFeature,
                       QgsGeometry,
                       QgsPointXY,
                       QgsField,
                       QgsFields,
                       QgsProcessing,
                       QgsProcessingException,
                       QgsProcessingUtils,
                       QgsFeatureSink,
                       QgsProcessingParameterDefinition,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterFeatureSink)

from processing.algs.qgis.QgisAlgorithm import QgisAlgorithm
from spatial_analysis.forms.KmeansWssParam import ParameterWss
import numpy as np
from scipy.cluster.vq import kmeans,vq

pluginPath = os.path.split(os.path.split(os.path.dirname(__file__))[0])[0]


class Kmeans(QgisAlgorithm):

    INPUT = 'INPUT_POINTS'
    K = 'K'
    MAX_K = 'MAX_K'
    WSS = 'WSS'
    OUTPUT = 'OUTPUT'
    OUTPUT_CENTROID = 'OUTPUT_CENTROID'
	
    def icon(self):
        return QIcon(os.path.join(pluginPath, 'spatial_analysis', 'icons', 'cluster.svg'))

    def group(self):
        return self.tr('Clustering')

    def groupId(self):
        return 'clustering'
    
    def name(self):
        return 'kmeans'

    def displayName(self):
        return self.tr('K-Means')
    
    def msg(self, var):
        return "Type:"+str(type(var))+" repr: "+var.__str__()

    def __init__(self):
        super().__init__()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(self.INPUT,
                                                              self.tr(u'Point Layer'),
                                                              [QgsProcessing.TypeVectorPoint]))
        self.addParameter(QgsProcessingParameterNumber(self.K,
                                                       self.tr(u'Number of Clusters(K)'),
                                                       QgsProcessingParameterNumber.Integer,
                                                       3, False, 2, 99999999))
        maxParam =QgsProcessingParameterNumber(self.MAX_K,
                                                       self.tr(u'Check WSS decrease pattern (enter "k" number of clusters)'),
                                                       QgsProcessingParameterNumber.Integer,
                                                       3, False, 2, 99999999)
        maxParam.setFlags(maxParam.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(maxParam)
        wss_param = ParameterWss(self.WSS, self.tr(u'Click "On Panel" or "On Browser"'), layer_param=self.INPUT, max_param=self.MAX_K)
        wss_param.setMetadata({'widget_wrapper': {'class': 'spatial_analysis.forms.KmeansWss.WssWidgetWrapper'}})
        wss_param.setFlags(wss_param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(wss_param)
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, 
                                                            self.tr(u'Output Layer with K_Clusters'),
                                                            QgsProcessing.TypeVectorPoint))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT_CENTROID, 
                                                            self.tr(u'Centroids of Clusters'),
                                                            QgsProcessing.TypeVectorPoint))

    def processAlgorithm(self, parameters, context, feedback):
        feedback.pushInfo(self.tr("Starting Algorithm: '{}'".format(self.displayName())))
        cLayer = self.parameterAsSource(parameters, self.INPUT, context)
        nCluster = self.parameterAsInt(parameters, self.K, context)		
        pts=[f.geometry().asPoint() for f in cLayer.getFeatures()]
        if nCluster > len(pts):
            feedback.pushInfo(self.tr(u'The number of clusters is greater than the number of data.<br>The number of clusters was adjusted to the number of data.'))
            nCluster = len(pts)

        # get coordinates of point features
        x=[pt[0] for pt in pts]
        y=[pt[1] for pt in pts]
        coords = np.stack([x, y], axis = -1)

        # assign cluster id, within cluster distance and centroid
        centroids,_ = kmeans(coords, nCluster)	
        cluster, distance = vq(coords, centroids)
		
        feedback.pushInfo("End of Algorithm")
        feedback.pushInfo("Building Layers")
		
        # cluster layer
        fields = cLayer.fields()
        new_fields = QgsFields()
        new_fields.append(QgsField('K_Cluster', QVariant.Int))
        new_fields.append(QgsField('Within_Cluster_D', QVariant.Double))
        fields = QgsProcessingUtils.combineFields(fields, new_fields)
        (cluster_sink, cluster_dest_id) = self.parameterAsSink(parameters, self.OUTPUT, context,
                                          fields, cLayer.wkbType(), cLayer.sourceCrs())

        total = len(coords)
        for i, feat in enumerate(cLayer.getFeatures()):
            outFeat = feat
            attrs = feat.attributes()
            attrs.extend([int(cluster[i])+1, float(distance[i])])
            outFeat.setAttributes(attrs)
            cluster_sink.addFeature(outFeat, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(i / total * 100))
        feedback.setProgress(0)
        feedback.pushInfo("Done with Cluster Layer")
		
        # cluster centroid layer
        xy_fields = QgsFields()
        xy_fields.append(QgsField('KCluster_ID', QVariant.Int))
        (centroid_sink, centroid_dest_id) = self.parameterAsSink(parameters, self.OUTPUT_CENTROID, context,
                                            xy_fields, cLayer.wkbType(), cLayer.sourceCrs())		

        centers = self.show_center(centroids)
        total = len(centers)
        for j, center in enumerate(centers):
            centerFeat = QgsFeature()
            centerGeom = QgsGeometry.fromPointXY(center)
            attrs = centerFeat.attributes()
            centerFeat.setGeometry(centerGeom)
            attrs.extend([int(j)+1])
            centerFeat.setAttributes(attrs)
            centroid_sink.addFeature(centerFeat, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(j / total * 100))
        feedback.pushInfo("Done with Cluster Centroid Layer")
        results = {}
        results[self.OUTPUT] = cluster_dest_id
        results[self.OUTPUT_CENTROID] = centroid_dest_id
        return results		

    def show_center(self, centroids):
        cluster_centers = [QgsPointXY(i[0],i[1]) for i in centroids]
        return cluster_centers
