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

from qgis.core import (QgsField,
                       QgsFields,
                       QgsProcessing,
                       QgsProcessingUtils,
                       QgsFeatureSink,
                       QgsProcessingParameterDefinition,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterFeatureSink)

from processing.algs.qgis.QgisAlgorithm import QgisAlgorithm
from spatial_analysis.forms.DbscanKnnParam import ParameterKnn
import processing
import numpy as np

pluginPath = os.path.split(os.path.split(os.path.dirname(__file__))[0])[0]


class Dbscan(QgisAlgorithm):

    INPUT = 'INPUT_POINTS'
    EPSILON = 'EPSILON'
    KNN = 'KNN'
    MINPOINTS = 'MINPOINTS'
    OUTPUT = 'OUTPUT'
	
    def icon(self):
        return QIcon(os.path.join(pluginPath, 'spatial_analysis', 'icons', 'cluster.svg'))

    def group(self):
        return self.tr('Clustering')

    def groupId(self):
        return 'clustering'
    
    def name(self):
        return 'dbscan'

    def displayName(self):
        return self.tr('DBSCAN')
    
    def msg(self, var):
        return "Type:"+str(type(var))+" repr: "+var.__str__()

    def __init__(self):
        super().__init__()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(self.INPUT,
                                                              self.tr(u'Point Layer'),
                                                              [QgsProcessing.TypeVectorPoint]))
        self.addParameter(QgsProcessingParameterNumber(self.EPSILON,
                                                       self.tr('Maximum Distance between Samples in a Cluster(ε)'),
                                                       QgsProcessingParameterNumber.Double,
                                                       0.1, False, 0, 99999999))
        self.addParameter(QgsProcessingParameterNumber(self.MINPOINTS,
                                                       self.tr(u'Minimum Cluster Size(minPts)'),
                                                       QgsProcessingParameterNumber.Integer,
                                                       3, False, 2, 99999999))
        knn_param = ParameterKnn(self.KNN, self.tr(u'<hr>Epsilon Distance to K Nearest Neighbors'), layer_param=self.INPUT, k_param=self.MINPOINTS)
        knn_param.setMetadata({'widget_wrapper': {'class': 'spatial_analysis.forms.DbscanKnn.KnnWidgetWrapper'}})
        self.addParameter(knn_param)                                                      
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, 
                                                            self.tr('Output Layer with DBSCAN'),
                                                            QgsProcessing.TypeVectorPoint))

    def processAlgorithm(self, parameters, context, feedback):
        feedback.pushInfo(self.tr("Starting Algorithm: '{}'".format(self.displayName())))
        cLayer = self.parameterAsSource(parameters, self.INPUT, context)
        epsilon = self.parameterAsDouble(parameters, self.EPSILON, context)
        minPts = self.parameterAsInt(parameters, self.MINPOINTS, context)		
        
        # get coordinates of point features
        pts=[f.geometry().asPoint() for f in cLayer.getFeatures()]            
        x=[pt[0] for pt in pts]
        y=[pt[1] for pt in pts]

        prs = {'INPUT':parameters[self.INPUT], 'MIN_SIZE':minPts, 'EPS':epsilon, 'OUTPUT':'memory:'}
        res = processing.run('native:dbscanclustering', prs, feedback=feedback)
        vl = res['OUTPUT']

        # assign cluster id
        clusterFieldIndex = cLayer.fields().count()
        cluster = [f.attributes()[clusterFieldIndex] for f in vl.getFeatures()]

        feedback.pushInfo("End of Algorithm")
        feedback.pushInfo("Building Layers")
		
        # cluster layer
        fields = cLayer.fields()
        new_fields = QgsFields()
        new_fields.append(QgsField('D_Cluster', QVariant.Int))
        fields = QgsProcessingUtils.combineFields(fields, new_fields)
        (cluster_sink, cluster_dest_id) = self.parameterAsSink(parameters, self.OUTPUT, context,
                                          fields, cLayer.wkbType(), cLayer.sourceCrs())

        total = len(pts)
        for i, feat in enumerate(cLayer.getFeatures()):
            outFeat = feat
            attrs = feat.attributes()
            attrs.extend([cluster[i]])
            outFeat.setAttributes(attrs)
            cluster_sink.addFeature(outFeat, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(i / total * 100))
        feedback.setProgress(0)
        feedback.pushInfo("Done with Cluster Layer")

        results = {}
        results[self.OUTPUT] = cluster_dest_id
        return results