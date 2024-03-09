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


class StandardDeviationEllipse(QgisAlgorithm):

    INPUT_POINTS = 'INPUT_POINTS'
    GROUP_FIELD = 'GROUP_FIELD'
    WEIGHT_FIELD = 'WEIGHT_FIELD'
    DF = 'DF'
    OUTPUT = 'OUTPUT'
	
    def icon(self):
        return QIcon(os.path.join(pluginPath, 'urban_analysis', 'icons', 'sde.png'))

    def group(self):
        return self.tr('Spatial Dispersion')

    def groupId(self):
        return 'spatialdispersion'
    
    def name(self):
        return 'standarddeviationellipse'

    def displayName(self):
        return self.tr('Standard Deviation Ellipse')
    
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
                                                        defaultValue=True))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, 
                                                            self.tr('Standard Deviation Ellipse'),
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

        # SDE layer
        sdeFields = QgsFields()
        if wFieldIndex > -1:
            sdeFields.append(QgsField('group', fields[groupFieldIndex].type()))
        else:
            sdeFields.append(QgsField('group', QVariant.Int))
        sdeFields.append(QgsField('meanx', QVariant.Double))
        sdeFields.append(QgsField('meany', QVariant.Double))
        sdeFields.append(QgsField("rotation", QVariant.Double))
        sdeFields.append(QgsField("sigmax", QVariant.Double))
        sdeFields.append(QgsField("sigmay", QVariant.Double))

        (sink, dest_id) = self.parameterAsSink(parameters, self.OUTPUT, context,
                                               sdeFields, QgsWkbTypes.Polygon, cLayer.sourceCrs())	
        if groupFieldIndex > -1 :
            for group in groupList:
                query = '"{field}" = {value}'.format(field = groupField, value = group)
                exp = QgsExpression(query)
                request = QgsFeatureRequest(exp)
                request.setSubsetOfAttributes([wFieldIndex])
                feat = [f for f in cLayer.getFeatures(request)]
                if len(feat) < 3:
                    continue
                sdeFeat = self.calc_sde(group, feat, weighted, wFieldIndex, dfCorrection)
                sink.addFeature(sdeFeat, QgsFeatureSink.FastInsert)
        else:
            request = QgsFeatureRequest()
            request.setSubsetOfAttributes([wFieldIndex])				
            feat = [f for f in cLayer.getFeatures(request)]
            sdeFeat = self.calc_sde(1, feat, weighted, wFieldIndex, dfCorrection)
            sink.addFeature(sdeFeat, QgsFeatureSink.FastInsert)

        feedback.pushInfo("Done!")
		
        results = {}
        results[self.OUTPUT] = dest_id
        return results		

    def calc_sde(self, group, feat, weighted, wFieldIndex, dfCorrection):
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

            vx=(x-mx)*(x-mx)*weights
            vy=(y-my)*(y-my)*weights
            vxy = (x-mx)*(y-my)*weights

            a = sum(vx) - sum(vy)
            b = sqrt(a*a + 4*sum(vxy)*sum(vxy)) 
            c = 2*sum(vxy)

            tantheta = (a+b) / c
            theta_d = degrees(atan(tantheta))+180*(tantheta<0)
            theta = radians(theta_d)
            sintheta=sin(theta)
            costheta=cos(theta)

            sigmax = sqrt(2) * sqrt(((sum(vx)) * (costheta * costheta) - 2 * (sum(vxy)) * (sintheta * costheta) + (sum(vy)) * (sintheta * sintheta))/((sum(weights)) - dfCorrection))
            sigmay = sqrt(2) * sqrt(((sum(vx)) * (sintheta * sintheta) + 2 * (sum(vxy)) * (sintheta * costheta) + (sum(vy)) * (costheta * costheta))/((sum(weights)) - dfCorrection))

            step = 360
            coords = []

            for i in range(step): 
                angle = i * 2 * pi / step 
                x1 = (sigmax) * cos(angle) 
                y1 = (sigmay) * sin(angle) 
                coords.append( QgsPointXY(x1, y1) ) 
				
            sdeFeat = QgsFeature()
            sdeGeom = QgsGeometry.fromPolygonXY([coords])
            sdeGeom.rotate(theta_d, QgsPointXY(0,0))
            sdeGeom.translate(mx, my)
            attrs = sdeFeat.attributes()
            sdeFeat.setGeometry(sdeGeom)
            attrs.extend([group, float(mx), float(my), float(theta_d), sigmax, sigmay])
            sdeFeat.setAttributes(attrs)
        except:
            pass
        else:
            return sdeFeat