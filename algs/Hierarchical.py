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
from spatial_analysis.forms.VariableParam import ParameterVariable

pluginPath = os.path.split(os.path.split(os.path.dirname(__file__))[0])[0]


class Hierarchical(QgisAlgorithm):

    INPUT = 'INPUT_LAYER'
    V_OPTIONS = 'V_OPTIONS'
    LINKAGE = 'LINKAGE'
    HARCHIPARAM = 'HARCHIPARAM'
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
                                                              self.tr(u'Input Layer'),
                                                              [QgsProcessing.TypeVector]))
        variable_param = ParameterVariable(self.V_OPTIONS, self.tr(u'Variable Fields'), layer_param=self.INPUT)
        variable_param.setMetadata({'widget_wrapper': {'class': 'spatial_analysis.forms.VariableWidget.VariableWidgetWrapper'}})
        self.addParameter(variable_param)
        self.dMethod = ['centroid', 'ward', 'single', 'complete', 'average']
        self.addParameter(QgsProcessingParameterEnum(self.LINKAGE,
                                                        self.tr('Linkage(Distance) Method'),
                                                        defaultValue  = 0,
                                                        options = self.dMethod))
        harchi_param = ParameterHarchi(self.HARCHIPARAM, self.tr(u'Clustering by'), layer_param=self.INPUT, variable_options=self.V_OPTIONS, linkage_param=self.LINKAGE)
        harchi_param.setMetadata({'widget_wrapper': {'class': 'spatial_analysis.forms.Harchi.HarchiWidgetWrapper'}})
        self.addParameter(harchi_param)
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, 
                                                            self.tr('Output Layer with H_Clusters'),
                                                            QgsProcessing.TypeVector))
    def processAlgorithm(self, parameters, context, feedback):
        feedback.pushInfo(self.tr("Starting Algorithm: '{}'".format(self.displayName())))
        cLayer = self.parameterAsSource(parameters, self.INPUT, context)
        to_cluster, variable_fields, normalized = self.parameterAsMatrix(parameters, self.V_OPTIONS, context)
        dMethodIndex = self.parameterAsEnum(parameters, self.LINKAGE, context)
        harchi_param = self.parameterAsMatrix(parameters, self.HARCHIPARAM, context)
        if harchi_param[0] == 0:
            criterion = 'maxclust'
        else:
            criterion = 'distance'
        threshold = harchi_param[1]

        # input --> numpy array
        if to_cluster == 'geom':
            features = [[f.geometry().centroid().asPoint().x(), f.geometry().centroid().asPoint().y()] for f in cLayer.getFeatures()]
            features = np.stack(features, axis = 0)
        else:
            features = [[f[fld] for f in cLayer.getFeatures()] for fld in variable_fields]
            features = np.stack(features, axis = 1)
        if normalized:
            features = whiten(features)
            if_normalized = "Yes"
        else:
            if_normalized = "No"
		
        ## perform hierarchical clustering and get centers of clusters
        Z = linkage(features, method = self.dMethod[dMethodIndex], metric = 'euclidean')
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
        total = len(features)
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
        