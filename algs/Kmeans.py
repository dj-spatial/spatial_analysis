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
import codecs

from qgis.PyQt.QtCore import QVariant, QUrl
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWebKitWidgets import QWebView

from qgis.core import (QgsWkbTypes,
                       QgsFeature,
                       QgsFeatureRequest,
                       QgsGeometry, QgsMessageLog,
                       QgsPointXY,
                       QgsField,
                       QgsFields,
                       QgsProcessing,
                       QgsProcessingException,
                       QgsProcessingUtils,
                       QgsFeatureSink,
                       QgsProcessingParameterDefinition,
                       QgsProcessingParameterEnum,
                       QgsProcessingParameterBoolean,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterField,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterFeatureSink,
                       QgsProcessingParameterFileDestination)

from processing.algs.qgis.QgisAlgorithm import QgisAlgorithm
from spatial_analysis.forms.KmeansWssParam import ParameterWss
from spatial_analysis.forms.VariableParam import ParameterVariable
import numpy as np
from scipy.cluster.vq import kmeans,vq, whiten, kmeans2
import geopandas as gpd

pluginPath = os.path.split(os.path.split(os.path.dirname(__file__))[0])[0]


class Kmeans(QgisAlgorithm):

    INPUT = 'INPUT_LAYER'
    MINIT = 'MINIT'
    ITER = 'ITER'
    K = 'K'
    V_OPTIONS = 'V_OPTIONS'
    NORMALIZE = 'NORMALIZE'
    WSS = 'WSS'
    OUTPUT = 'OUTPUT'
    OUTPUT_CENTROID = 'OUTPUT_CENTROID'
    OUTPUT_REPORT = 'OUTPUT_REPORT'
	
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

    """
    def shortHelpString(self):
        return self.tr("Example <br> script with a custom widget")
    """
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
        self.minit_name=[['KMeans++', 'Random', 'Points'], ['++', 'random', 'points']]
        self.addParameter(QgsProcessingParameterEnum(self.MINIT,
                                                     self.tr(u'Initialization Method'),
                                                     defaultValue  = 0,
                                                     options = self.minit_name[0]))
        self.addParameter(QgsProcessingParameterNumber(self.ITER,
                                                       self.tr(u'Number of iterations'),
                                                       QgsProcessingParameterNumber.Integer,
                                                       10, False, 1, 99999999))
        self.addParameter(QgsProcessingParameterNumber(self.K,
                                                       self.tr(u'Number of Clusters(K)'),
                                                       QgsProcessingParameterNumber.Integer,
                                                       3, False, 2, 99999999))
        wss_param = ParameterWss(self.WSS, self.tr('<hr>Elbow Graph'), layer_param=self.INPUT, variable_options=self.V_OPTIONS)
        wss_param.setMetadata({'widget_wrapper': {'class': 'spatial_analysis.forms.KmeansWss.WssWidgetWrapper'}})
        self.addParameter(wss_param)
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, 
                                                            self.tr(u'Output Layer with K_Clusters'),
                                                            QgsProcessing.TypeVector))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT_CENTROID, 
                                                            self.tr(u'Centroids of Clusters'),
                                                            QgsProcessing.TypeVectorPoint))
        self.addParameter(QgsProcessingParameterFileDestination(self.OUTPUT_REPORT, self.tr('Output Report'),
                                                                self.tr('HTML files (*.html)'), None, True))

    def processAlgorithm(self, parameters, context, feedback):
        # input parameters
        cLayer = self.parameterAsSource(parameters, self.INPUT, context)
        to_cluster, variable_fields, normalized = self.parameterAsMatrix(parameters, self.V_OPTIONS, context)
        minit_idx = self.parameterAsEnum(parameters, self.MINIT, context)
        iter = self.parameterAsInt(parameters, self.ITER, context)
        k = self.parameterAsInt(parameters, self.K, context)
        feat_count = len(cLayer)
        if k > feat_count:
            feedback.pushInfo(self.tr(u'The number of clusters is greater than the number of data.<br>The number of clusters was adjusted to the number of data.'))
            k = feat_count
        if to_cluster == 'attrs' and not variable_fields:
            raise QgsProcessingException(self.tr(u'No Fields Selected.'))
        feedback.pushInfo(self.tr("Starting Algorithm: '{}'".format(self.displayName())))

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
        total_ss = np.sum((features - np.mean(features, axis = 0))**2)
        features = np.insert(features, 0, range(0, feat_count), axis = 1) # add index

        # k means clustering
        centroids, label = kmeans2(features[:, 1:], k, iter, minit=self.minit_name[1][minit_idx])
        features = np.insert(features, 1, label, axis = 1) # add cluster id
        centroids =  np.insert(centroids, 0, range(0, k), axis = 1) # add cluster id

        pts = []
        wss_by_cluster = []
        for cluster_id in range(0, k):
            chunk = features[features[:, 1]==cluster_id]
            distance_squared = np.sum((chunk[:, 2:] - centroids[cluster_id, 1:])**2, axis=1)
            chunk = np.insert(chunk, 2, distance_squared, axis=1)
            pts.append(chunk)
            wss_by_cluster.append(np.sum(distance_squared))
        pts = np.vstack(pts)
        pts = pts[np.argsort(pts[:, 0]), :] #re-order by index(1st column)


        centroids = np.insert(centroids, 1, wss_by_cluster, axis=1) # add wss
        cluster = pts[:,1]
        distance = pts[:,2]
        
        feedback.pushInfo("End of Algorithm")
        feedback.pushInfo("Building Layers")
	
        # cluster layer
        fields = cLayer.fields()
        new_fields = QgsFields()
        new_fields.append(QgsField('K_Cluster_ID', QVariant.Int))
        new_fields.append(QgsField('Within_Distance', QVariant.Double))
        fields = QgsProcessingUtils.combineFields(fields, new_fields)
        (cluster_sink, cluster_dest_id) = self.parameterAsSink(parameters, self.OUTPUT, context,
                                          fields, cLayer.wkbType(), cLayer.sourceCrs())
        for i, feat in enumerate(cLayer.getFeatures()):
            outFeat = feat
            attrs = feat.attributes()
            attrs.extend([int(cluster[i]), float(distance[i])])
            outFeat.setAttributes(attrs)
            cluster_sink.addFeature(outFeat, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(i / feat_count * 100))
        feedback.setProgress(0)
        feedback.pushInfo("Done with Cluster Layer")
		
        # centroid layer
        if to_cluster == 'geom':
            xy_fields = QgsFields()
            xy_fields.append(QgsField('K_Cluster_ID', QVariant.Int))
            xy_fields.append(QgsField('WSS', QVariant.Double))
            (centroid_sink, centroid_dest_id) = self.parameterAsSink(parameters, self.OUTPUT_CENTROID, context,
                                                xy_fields, QgsWkbTypes.Point, cLayer.sourceCrs())		

            total = k
            for j, center in enumerate(centroids):
                centerFeat = QgsFeature()
                if to_cluster == 'geom':
                    centerGeom = QgsGeometry.fromPointXY(QgsPointXY(center[2],center[3]))
                    centerFeat.setGeometry(centerGeom)
                attrs = centerFeat.attributes()
                attrs.extend([int(center[0]), float(center[1])])
                centerFeat.setAttributes(attrs)
                centroid_sink.addFeature(centerFeat, QgsFeatureSink.FastInsert)
                feedback.setProgress(int(j / total * 100))
            feedback.pushInfo(self.tr("Done with Cluster <br> Centroid Layer"))
        else:
            centroid_sink = None
            centroid_dest_id = None

        # output report
        output_report = self.parameterAsFileOutput(parameters, self.OUTPUT_REPORT, context)

        total_wss = np.sum(centroids[:, 1])
        td_blue = '<td rowspan="1" \
                    colspan="1" \
                    bgcolor="rgb(0, 80, 141)" \
                    style="word-break: break-all; background-color: rgb(0, 80, 141); \
                    height: 24px; \
                    padding: 3px 4px 2px;" \
                    data-origin-bgcolor="rgb(0, 80, 141)">' + \
                    '<div style="text-align: center;">' + \
                    '<span style="color: rgb(255, 255, 255); font-weight: bold;">'
        td_white = '<td rowspan="1" \
                    colspan="1" \
                    bgcolor="#ffffff" \
                    style="word-break: break-all; background-color: rgb(255, 255, 255); \
                    height: 24px; \
                    padding: 3px 4px 2px;">' + \
                    '<div style="text-align: center;">' + \
                    '<span>'

        with codecs.open(output_report, 'w', encoding='utf-8') as f:
            f.write('<html><head>\n')
            f.write('<meta http-equiv="Content-Type" content="text/html; \
                    charset=utf-8" /></head><body>\n')
            f.write('<p> Number of clusters: ' + str(k) + '</p>\n')
            f.write('<p> Initialization Method: ' + self.minit_name[0][minit_idx] + '</p>\n')
            f.write('<p> Number of iterations: ' + str(iter) + '</p>\n')
            f.write('<p> Normalized: ' + if_normalized + '</p>\n')
            f.write('<p> The total sum of squares: ' + str(total_ss) + '</p>\n')
            f.write('<p> The within cluster sum of squares: ' + str(total_wss) + '</p>\n')
            f.write('<p> The between cluster sum of squares: ' + str(total_ss-total_wss) + '</p>\n')
            f.write('<p> The ratio of between to total sum of squares: ' + str((total_ss-total_wss) / total_ss * 100) + '%</p>\n')
            # start of table
            f.write('<table cellpadding="0" cellspacing="1" bgcolor="#ffffff" style="background-color: rgb(204, 204, 204);">')
            f.write('<tbody>')
            f.write('<tr style="">')
            f.write(td_blue + 'Cluster Centers' + '</span></div></td>')
            f.write(td_blue + 'WSS' + '</span></div></td>')
            
            cols = ['X', 'Y'] if to_cluster == 'geom' else variable_fields
            for c in cols:
                f.write(td_blue + str(c) + '</span></div></td>')
                
            for centroid in centroids:
                f.write('<tr style="height: 24px;">')
                for cent in centroid:
                    f.write(td_white + str(cent) + '</span></div></td>')
                f.write('<tr>')
            f.write('</tbody></table>')
            f.write('</body></html>\n')

        results = {}
        results[self.OUTPUT] = cluster_dest_id
        results[self.OUTPUT_CENTROID] = centroid_dest_id
        results[self.OUTPUT_REPORT] = output_report
        return results		
