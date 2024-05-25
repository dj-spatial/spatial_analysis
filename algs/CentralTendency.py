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
from qgis.core import (QgsProcessingParameterBoolean,
                       QgsExpression,
                       QgsFeatureRequest,
                       QgsField,
                       QgsFields,
                       QgsPointXY,
                       QgsFeatureSink,
                       QgsProcessing,
                       QgsProcessingParameterField,
                       QgsProcessingParameterEnum,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterFeatureSink)

from processing.algs.qgis.QgisAlgorithm import QgisAlgorithm
from ..utilities import getMeanCenter
from ..utilities import getMedianCenter
from ..utilities import getCentralFeature
from ..utilities import getPointCoords

pluginPath = os.path.split(os.path.split(os.path.dirname(__file__))[0])[0]


class CentralTendency(QgisAlgorithm):

    INPUT_POINTS = 'INPUT_POINTS'
    GROUP_FIELD = 'GROUP_FIELD'
    WEIGHT_FIELD = 'WEIGHT_FIELD'
    MEAN_CENTER = 'MEAN_CENTER'
    MEDIAN_CENTER = 'MEDIAN_CENTER'
    CENTRAL_FEATURE = 'CENTRAL_FEATURE'
    METRIC = 'METRIC'
    OUTPUT_MEAN = 'OUTPUT_MEAN'
    OUTPUT_MEDIAN = 'OUTPUT_MEDIAN'
    OUTPUT_CENTRAL = 'OUTPUT_CENTRAL'
	
    def icon(self):
        return QIcon(os.path.join(pluginPath, 'spatial_analysis', 'icons', 'central.svg'))

    def group(self):
        return self.tr('Spatial Central Tendency')

    def groupId(self):
        return 'centraltendency'
    
    def name(self):
        return 'spatialcenters'

    def displayName(self):
        return self.tr('Centers(Mean Center, Median Center, Central Feature)')
    
    def msg(self, var):
        return "Type:"+str(type(var))+" repr: "+var.__str__()

    def __init__(self):
        super().__init__()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(self.INPUT_POINTS,
                                                              self.tr(u'Point Layer'),
                                                              [QgsProcessing.TypeVectorPoint]))
        self.addParameter(QgsProcessingParameterField(self.GROUP_FIELD,
                                                      self.tr('Group Field'),
                                                      parentLayerParameterName=self.INPUT_POINTS,
                                                      type=QgsProcessingParameterField.Any, optional=True))
        self.addParameter(QgsProcessingParameterField(self.WEIGHT_FIELD,
                                                      self.tr(u'Weight Field'),
                                                      parentLayerParameterName=self.INPUT_POINTS,
                                                      type=QgsProcessingParameterField.Numeric, optional=True))
        self.addParameter(QgsProcessingParameterBoolean(self.MEAN_CENTER,
                                                        self.tr(u'Mean Center'),
                                                        defaultValue = True))
        self.addParameter(QgsProcessingParameterBoolean(self.MEDIAN_CENTER,
                                                        self.tr(u'Median Center'),
                                                        defaultValue = True))
        self.addParameter(QgsProcessingParameterBoolean(self.CENTRAL_FEATURE,
                                                        self.tr(u'Central Feature'),
                                                        defaultValue = True))
        self.addParameter(QgsProcessingParameterEnum(self.METRIC,
                                                        self.tr(u'Distance Metric(Only for Central Feature)'),
                                                        defaultValue  = 0,
                                                        options = ['Euclidean', 'City Block']))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT_MEAN, 
                                                            self.tr(u'MEAN_CENTER'),
                                                            QgsProcessing.TypeVectorPoint))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT_MEDIAN, 
                                                            self.tr(u'MEDIAN_CENTER'),
                                                            QgsProcessing.TypeVectorPoint))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT_CENTRAL, 
                                                            self.tr(u'CENTRAL_FEATURE'),
                                                            QgsProcessing.TypeVectorPoint))

    def processAlgorithm(self, parameters, context, feedback):
        feedback.pushInfo(self.tr("Starting Algorithm: '{}'".format(self.displayName())))
        cLayer = self.parameterAsSource(parameters, self.INPUT_POINTS, context)
        fields = cLayer.fields()
        groupField = self.parameterAsString(parameters, self.GROUP_FIELD, context)
        groupFieldIndex = fields.lookupField(self.parameterAsString(parameters, self.GROUP_FIELD, context))
        groupList = sorted(cLayer.uniqueValues(groupFieldIndex))

        meanCenter = self.parameterAsBool(parameters, self.MEAN_CENTER, context)
        medianCenter = self.parameterAsBool(parameters, self.MEDIAN_CENTER, context)
        centralFeature = self.parameterAsBool(parameters, self.CENTRAL_FEATURE, context)
        dMetricIndex = self.parameterAsEnum(parameters, self.METRIC, context)

        weightFieldIndex = fields.lookupField(self.parameterAsString(parameters, self.WEIGHT_FIELD, context))

        # create an id field
        idField = QgsFields()
        idField.append(QgsField('ID', QVariant.Int))
        results = {}

        if  meanCenter:
            (sink_mean, mean_id) = self.parameterAsSink(parameters, self.OUTPUT_MEAN, context,
                                                        idField, cLayer.wkbType(), cLayer.sourceCrs())
            if groupFieldIndex > -1 :
                for group in groupList:
                    query = '"{field}" = {value}'.format(field = groupField, value = group)
                    exp = QgsExpression(query)
                    request = QgsFeatureRequest(exp)
                    feat = [f for f in cLayer.getFeatures(request)]
                    pointCoords = getPointCoords(feat, weightFieldIndex)
                    meanCenterFeat = getMeanCenter(pointCoords[0], pointCoords[1], pointCoords[2], 1)
                    sink_mean.addFeature(meanCenterFeat, QgsFeatureSink.FastInsert)
            else:
                feat = cLayer.getFeatures()
                pointCoords = getPointCoords(feat, weightFieldIndex)
                meanCenterFeat = getMeanCenter(pointCoords[0], pointCoords[1], pointCoords[2], 1)
                sink_mean.addFeature(meanCenterFeat, QgsFeatureSink.FastInsert)
            results[self.OUTPUT_MEAN] = mean_id

        if  medianCenter:
            (sink_median, median_id) = self.parameterAsSink(parameters, self.OUTPUT_MEDIAN, context,
                                                            idField, cLayer.wkbType(), cLayer.sourceCrs())
            if groupFieldIndex > -1 :
                for group in groupList:
                    query = '"{field}" = {value}'.format(field = groupField, value = group)
                    exp = QgsExpression(query)
                    request = QgsFeatureRequest(exp)
                    feat = [f for f in cLayer.getFeatures(request)]
                    pointCoords = getPointCoords(feat, weightFieldIndex)
                    medianCenterFeat = getMedianCenter(pointCoords[0], pointCoords[1], pointCoords[2], 1)
                    sink_median.addFeature(medianCenterFeat, QgsFeatureSink.FastInsert)
                results[self.OUTPUT_MEDIAN] = median_id
            else:
                feat = cLayer.getFeatures()
                pointCoords = getPointCoords(feat, weightFieldIndex)
                medianCenterFeat = getMedianCenter(pointCoords[0], pointCoords[1], pointCoords[2], 1)
                sink_median.addFeature(medianCenterFeat, QgsFeatureSink.FastInsert)   
            results[self.OUTPUT_MEDIAN] = median_id                    

        if  centralFeature:
            (sink_central, central_id) = self.parameterAsSink(parameters, self.OUTPUT_CENTRAL, context,
                                                              idField, cLayer.wkbType(), cLayer.sourceCrs())
            if groupFieldIndex > -1 :
                for group in groupList:
                    query = '"{field}" = {value}'.format(field = groupField, value = group)
                    exp = QgsExpression(query)
                    request = QgsFeatureRequest(exp)
                    feat = [f for f in cLayer.getFeatures(request)]
                    pointCoords = getPointCoords(feat, weightFieldIndex)
                    centralFeatureFeat = getCentralFeature(pointCoords[0], pointCoords[1], pointCoords[2], 1, dMetricIndex)
                    sink_central.addFeature(centralFeatureFeat, QgsFeatureSink.FastInsert)
            else:
                feat = cLayer.getFeatures()
                pointCoords = getPointCoords(feat, weightFieldIndex)
                centralFeatureFeat = getCentralFeature(pointCoords[0], pointCoords[1], pointCoords[2], 1, dMetricIndex)
                sink_central.addFeature(centralFeatureFeat, QgsFeatureSink.FastInsert)
            results[self.OUTPUT_CENTRAL] = central_id
        return results