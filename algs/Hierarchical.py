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
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterEnum,
                       QgsProcessingParameterDefinition,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterFeatureSink)

from processing.algs.qgis.QgisAlgorithm import QgisAlgorithm
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
from spatial_analysis.forms.HarchiParam import ParameterHarchi

pluginPath = os.path.split(os.path.split(os.path.dirname(__file__))[0])[0]


class Hierarchical(QgisAlgorithm):

    INPUT = 'INPUT_POINTS'
    LINKAGE = 'LINKAGE'
    HARCHIPARAM = 'HARCHIPARAM'
    CLUSTER_METHOD = 'CLUSTER_METHOD'
    MAX_K = 'MAX_K'
    THRESHOLD = 'THRESHOLD'
    DEPTH= 'DEPTH'
    OUTPUT = 'OUTPUT'
	
    def icon(self):
        return QIcon(os.path.join(pluginPath, 'spatial_analysis', 'icons', 'cluster.svg'))

    def group(self):
        return self.tr('Clustering')

    def groupId(self):
        return 'clustering'
    
    def name(self):
        return 'hierarchical'

    def displayName(self):
        return self.tr('Hierarchical')

    def msg(self, var):
        return "Type:"+str(type(var))+" repr: "+var.__str__()

    def __init__(self):
        super().__init__()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(self.INPUT,
                                                              self.tr(u'Point Layer'),
                                                              [QgsProcessing.TypeVectorPoint]))
        self.dMethod = ['centroid', 'ward', 'single', 'complete', 'average']
        self.addParameter(QgsProcessingParameterEnum(self.LINKAGE,
                                                        self.tr('Linkage(Distance) Method'),
                                                        defaultValue  = 0,
                                                        options = self.dMethod))
        harchi_param = ParameterHarchi(self.HARCHIPARAM, self.tr(u'Choose Graph Type'), layer_param=self.INPUT, linkage_param=self.LINKAGE)
        harchi_param.setMetadata({'widget_wrapper': {'class': 'spatial_analysis.forms.Harchi.HarchiWidgetWrapper'}})
        harchi_param.setFlags(harchi_param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(harchi_param)

        self.cMethod = [u'Cluster Numbers(User Defined)', u'Distance Based(Cut Tree)']
        self.addParameter(QgsProcessingParameterEnum(self.CLUSTER_METHOD,
                                                       self.tr('Cluster Method'),
                                                       defaultValue  = 0,
                                                       options = self.cMethod))
        self.addParameter(QgsProcessingParameterNumber(self.MAX_K,
                                                       self.tr(u' ▶ Cluster Number - Incase of *Cluster Numbers*'),
                                                       QgsProcessingParameterNumber.Integer,
                                                       3, False, 2, 99999999))
        self.addParameter(QgsProcessingParameterNumber(self.THRESHOLD,
                                                       self.tr(u' ▶ Distance - Incase of *Distance Based*'),
                                                       QgsProcessingParameterNumber.Double,
                                                       0, False, 0, 99999999))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, 
                                                            self.tr('Output Layer with H_Clusters'),
                                                            QgsProcessing.TypeVectorPoint))
    def processAlgorithm(self, parameters, context, feedback):
        feedback.pushInfo(self.tr("Starting Algorithm: '{}'".format(self.displayName())))
        cLayer = self.parameterAsSource(parameters, self.INPUT, context)
        dMethodIndex = self.parameterAsEnum(parameters, self.LINKAGE, context)
        cMethodIndex = self.parameterAsEnum(parameters, self.CLUSTER_METHOD, context)
        if cMethodIndex == 0:
            criterion = 'maxclust'
            threshold = self.parameterAsInt(parameters, self.MAX_K, context)
        else:
            criterion = 'distance'
            threshold = self.parameterAsDouble(parameters, self.THRESHOLD, context)

        ## get coordinates of point features
        pts=[f.geometry().asPoint() for f in cLayer.getFeatures()]            
        x=[pt[0] for pt in pts]
        y=[pt[1] for pt in pts]
        coords = np.stack([x, y], axis = -1)
		
        ## perform hierarchical clustering and get centers of clusters
        Z = linkage(coords, method = self.dMethod[dMethodIndex], metric = 'euclidean')
        # cluster = fcluster(Z, t = threshold, criterion = self.cMethod[cMethodIndex])
        cluster = fcluster(Z, t = threshold, criterion = criterion)
        feedback.pushInfo("End of Algorithm")
        feedback.pushInfo("Building Layers")
		
        ## cluster layer
        fields = cLayer.fields()
        new_fields = QgsFields()
        new_fields.append(QgsField('H_Cluster', QVariant.Int))

        fields = QgsProcessingUtils.combineFields(fields, new_fields)
        (cluster_sink, cluster_dest_id) = self.parameterAsSink(parameters, self.OUTPUT, context,
                                          fields, cLayer.wkbType(), cLayer.sourceCrs())
        total = len(coords)
        for i, feat in enumerate(cLayer.getFeatures()):
            outFeat = feat
            attrs = feat.attributes()
            attrs.extend([int(cluster[i])])
            outFeat.setAttributes(attrs)
            cluster_sink.addFeature(outFeat, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(i / total * 100))
        feedback.setProgress(0)
        feedback.pushInfo("Done with Cluster Layer")
   
        results = {}
        results[self.OUTPUT] = cluster_dest_id
        return results		
        