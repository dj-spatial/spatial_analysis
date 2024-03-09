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

from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtGui import QIcon

from qgis.core import (QgsExpression,
                       QgsFeatureRequest,
                       QgsField,
                       QgsFields,
                       QgsFeatureSink,
                       QgsProcessing,
                       QgsProcessingParameterField,
                       QgsProcessingParameterEnum,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterFeatureSink)

from processing.algs.qgis.QgisAlgorithm import QgisAlgorithm
from ..utilities import getCentralFeature
from ..utilities import getPointCoords

pluginPath = os.path.split(os.path.split(os.path.dirname(__file__))[0])[0]


class CentralFeatureTracker(QgisAlgorithm):

    INPUT_POINTS = 'INPUT_POINTS'
    METRIC = 'METRIC'
    START_FIELD = 'START_FIELD'
    END_FIELD = 'END_FIELD'
    WEIGHT_FIELD = 'WEIGHT_FIELD'
    OUTPUT = 'OUTPUT'
	
    def icon(self):
        return QIcon(os.path.join(pluginPath, 'urban_analysis', 'icons', 'tracker.png'))

    def group(self):
        return self.tr('Spatial Central Tendency')

    def groupId(self):
        return 'centraltendency'
    
    def name(self):
        return 'centralfeaturetracker'

    def displayName(self):
        return self.tr(u'누적중심피쳐')
    
    def msg(self, var):
        return "Type:"+str(type(var))+" repr: "+var.__str__()

    def __init__(self):
        super().__init__()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(self.INPUT_POINTS,
                                                              self.tr(u'포인트 레이어'),
                                                              [QgsProcessing.TypeVectorPoint]))
        self.addParameter(QgsProcessingParameterEnum(self.METRIC,
                                                        self.tr(u'거리단위'),
                                                        options = ['Euclidean', 'City Block']))
        self.addParameter(QgsProcessingParameterField(self.START_FIELD,
                                                       self.tr(u'시작 필드 - 숫자 또는 시간 형식'),
                                                       parentLayerParameterName=self.INPUT_POINTS,
                                                       type=QgsProcessingParameterField.Any))
        self.addParameter(QgsProcessingParameterField(self.END_FIELD,
                                                       self.tr(u'종료 필드 - 시작 필드와 동일 형식'),
                                                       parentLayerParameterName=self.INPUT_POINTS,
                                                       type=QgsProcessingParameterField.Any, optional=True))
        self.addParameter(QgsProcessingParameterField(self.WEIGHT_FIELD,
                                                      self.tr(u'가중치 필드'),
                                                      parentLayerParameterName=self.INPUT_POINTS,
                                                      type=QgsProcessingParameterField.Numeric, optional=True))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, 
                                                            self.tr(u'누적중심피쳐'),
                                                            QgsProcessing.TypeVectorPoint))

    def processAlgorithm(self, parameters, context, feedback):
        feedback.pushInfo(self.tr("Starting Algorithm: '{}'".format(self.displayName())))
        cLayer = self.parameterAsSource(parameters, self.INPUT_POINTS, context)
        dMetricIndex = self.parameterAsEnum(parameters, self.METRIC, context)
        fields = cLayer.fields()
        startField = self.parameterAsString(parameters, self.START_FIELD, context)
        startFieldIndex = fields.lookupField(startField)
        startFieldType = fields[startFieldIndex].type()
        startFieldTypeName = fields[startFieldIndex].typeName()
        endField = self.parameterAsString(parameters, self.END_FIELD, context)
        endFieldIndex = fields.lookupField(endField)
        timeStamps = sorted(cLayer.uniqueValues(startFieldIndex))
        weightFieldIndex = fields.lookupField(self.parameterAsString(parameters, self.WEIGHT_FIELD, context))

        # central feature layer
        timeStampFields = QgsFields()
        timeStampFields.append(QgsField('TIME Stamp',fields[startFieldIndex].type()))
        (sink, dest_id) = self.parameterAsSink(parameters, self.OUTPUT, context,
                                               timeStampFields, cLayer.wkbType(), cLayer.sourceCrs())
        total = len(timeStamps)
        for j, timeStamp in enumerate(timeStamps):
            feedback.setProgress(int(j / total * 100))
            if endFieldIndex > -1:
                if startFieldTypeName == 'Date' or startFieldTypeName == 'Time':
                    query = '"{sfield}" <= {value} and "{efield}" >= {value}'.format(sfield = startField, efield = endField, value = "'"+timeStamp.toString('yyyy-MM-dd')+"'")
                else:
                    query = '"{sfield}" <= {value} and "{efield}" >= {value}'.format(sfield = startField, efield = endField, value = timeStamp)
            else:
                if startFieldTypeName == 'Date' or startFieldTypeName == 'Time':
                    query = '"{sfield}" <= {value}'.format(sfield = startField, value = "'"+timeStamp.toString('yyyy-MM-dd')+"'")
                else:
                    query = '"{sfield}" <= {value}'.format(sfield = startField, value = timeStamp)

            # central features by time stamp
            expr = QgsExpression(query)
            feat = cLayer.getFeatures(QgsFeatureRequest(expr))
            pointCoords = getPointCoords(feat, weightFieldIndex)
            centralFeat = getCentralFeature(pointCoords[0], pointCoords[1], pointCoords[2], timeStamp, dMetricIndex)
            if centralFeat is None:
                continue
            sink.addFeature(centralFeat, QgsFeatureSink.FastInsert)
        feedback.setProgress(100)
        feedback.pushInfo("Done")
		
        results = {}
        results[self.OUTPUT] = dest_id
        return results