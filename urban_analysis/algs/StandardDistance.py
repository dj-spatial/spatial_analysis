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
                       QgsFeature,
                       QgsGeometry,
                       QgsPointXY,
                       QgsField,
                       QgsFields,
                       QgsWkbTypes,
                       QgsFeatureSink,
                       QgsProcessing,
                       QgsProcessingParameterField,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterBoolean,
                       QgsProcessingParameterFeatureSink)

from processing.algs.qgis.QgisAlgorithm import QgisAlgorithm
import numpy as np
from math import degrees, radians, sqrt, pow, sin, cos, atan, pi

pluginPath = os.path.split(os.path.split(os.path.dirname(__file__))[0])[0]


class StandardDistance(QgisAlgorithm):

    INPUT_POINTS = 'INPUT_POINTS'
    GROUP_FIELD = 'GROUP_FIELD'
    WEIGHT_FIELD = 'WEIGHT_FIELD'
    DF = 'DF'
    OUTPUT = 'OUTPUT'
	
    def icon(self):
        return QIcon(os.path.join(pluginPath, 'urban_analysis', 'icons', 'sdd.png'))

    def group(self):
        return self.tr('Spatial Dispersion')

    def groupId(self):
        return 'spatialdispersion'
    
    def name(self):
        return 'standarddistance'

    def displayName(self):
        return self.tr('Standard Distance')
    
    def msg(self, var):
        return "Type:"+str(type(var))+" repr: "+var.__str__()

    def __init__(self):
        super().__init__()


    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(self.INPUT_POINTS,
                                                              self.tr('Points Layer'),
                                                              [QgsProcessing.TypeVectorPoint]))
        self.addParameter(QgsProcessingParameterField(self.GROUP_FIELD,
                                                      self.tr('Group Field'),
                                                      parentLayerParameterName=self.INPUT_POINTS,
                                                      type=QgsProcessingParameterField.Any, optional=True))
        self.addParameter(QgsProcessingParameterField(self.WEIGHT_FIELD,
                                                      self.tr('Weight Field'),
                                                      parentLayerParameterName=self.INPUT_POINTS,
                                                      type=QgsProcessingParameterField.Numeric, optional=True))
        self.addParameter(QgsProcessingParameterBoolean(self.DF,
                                                        self.tr('DF Correction'),
                                                        defaultValue=False))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, 
                                                            self.tr('Standard Distance'),
                                                            QgsProcessing.TypeVectorPolygon))

    def processAlgorithm(self, parameters, context, feedback):
        feedback.pushInfo(self.tr("Starting Algorithm: '{}'".format(self.displayName())))
        cLayer = self.parameterAsSource(parameters, self.INPUT_POINTS, context)
        fields = cLayer.fields()
        groupField = self.parameterAsString(parameters, self.GROUP_FIELD, context)
        groupFieldIndex = fields.lookupField(self.parameterAsString(parameters, self.GROUP_FIELD, context))
        groupList = sorted(cLayer.uniqueValues(groupFieldIndex))
		
        # weight field
        wFieldIndex = fields.lookupField(self.parameterAsString(parameters, self.WEIGHT_FIELD, context))
        if wFieldIndex > -1:
            weighted = True 
            wFieldIndex = fields.lookupField(self.parameterAsString(parameters, self.WEIGHT_FIELD, context))
        else:
            weighted = False 

        # degree of freedom
        df = self.parameterAsBool(parameters, self.DF, context)
        if df == True:
            dfCorrection = 2 
        else:
            dfCorrection = 0 

        # SDD layer
        sddFields = QgsFields()
        if wFieldIndex > -1:
            sddFields.append(QgsField('group', fields[groupFieldIndex].type()))
        else:
            sddFields.append(QgsField('group', QVariant.Int))
        sddFields.append(QgsField('meanx', QVariant.Double))
        sddFields.append(QgsField('meany', QVariant.Double))
        sddFields.append(QgsField('SDD', QVariant.Double))
        (sink, dest_id) = self.parameterAsSink(parameters, self.OUTPUT, context,
                                               sddFields, QgsWkbTypes.Polygon, cLayer.sourceCrs())	
        if groupFieldIndex > -1 :
            for group in groupList:
                query = '"{field}" = {value}'.format(field = groupField, value = group)
                exp = QgsExpression(query)
                request = QgsFeatureRequest(exp)
                request.setSubsetOfAttributes([wFieldIndex])
                feat = [f for f in cLayer.getFeatures(request)]
                if len(feat) < 3:
                    continue
                sddFeat = self.calc_sdd(group, feat, weighted, wFieldIndex, dfCorrection)
                sink.addFeature(sddFeat, QgsFeatureSink.FastInsert)
        else:
            request = QgsFeatureRequest()
            request.setSubsetOfAttributes([wFieldIndex])				
            feat = [f for f in cLayer.getFeatures(request)]
            sddFeat = self.calc_sdd(1, feat, weighted, wFieldIndex, dfCorrection)
            sink.addFeature(sddFeat, QgsFeatureSink.FastInsert)

        feedback.pushInfo("Done!")
		
        results = {}
        results[self.OUTPUT] = dest_id
        return results		

    def calc_sdd(self, group, feat, weighted, wFieldIndex, dfCorrection):
        try:
            x=[]; y=[]; wx=[]; wy=[]
            if not weighted:
                weights = [1] * len(feat)
            else:
                weights = [f.attributes()[wFieldIndex] for f in feat]

            weights = np.asarray(weights, dtype = np.float32)

            for f in feat:
                geom = f.geometry()
                x.append(geom.asPoint().x())
                y.append(geom.asPoint().y())

            wx = x * weights
            wy = y * weights

            mx=sum(wx)/sum(weights)
            my=sum(wy)/sum(weights)

            dist = (x - mx)**2 + (y - my)**2
            sdd = sqrt(sum((weights*dist)/((sum(weights))-dfCorrection)))

            step = 360
            coords = []

            for i in range(step): 
                angle = i * 2 * pi / step 
                x1 = (sdd) * cos(angle) 
                y1 = (sdd) * sin(angle) 
                coords.append( QgsPointXY(x1, y1) ) 
				
            sddFeat = QgsFeature()
            sddGeom = QgsGeometry.fromPolygonXY([coords])
            sddGeom.translate(mx, my)
            attrs = sddFeat.attributes()
            sddFeat.setGeometry(sddGeom)
            attrs.extend([group, float(mx), float(my), sdd])
            sddFeat.setAttributes(attrs)
        except:
            pass
        else:
            return sddFeat