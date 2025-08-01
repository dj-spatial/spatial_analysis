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
__date__ = 'July 2024'
__copyright__ = '(C) 2024, D.J Paek'

# This will get replaced with a git SHA1 when you do a git archive

__revision__ = '$Format:%H$'

import os

from qgis.PyQt.QtCore import QVariant, QUrl
from qgis.PyQt.QtGui import QIcon

from qgis.core import (
    QgsField,
    QgsFields,
    QgsProcessing,
    QgsProcessingException,
    QgsProcessingUtils,
    QgsFeatureSink,
    QgsProcessingParameterField,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFeatureSink
)

from processing.algs.qgis.QgisAlgorithm import QgisAlgorithm

# pysal modules - these may not be available in all environments
try:
    from libpysal.weights import KNN
    from esda.getisord import G_Local
except Exception as e:  # pragma: no cover - library might be missing
    KNN = None
    G_Local = None

pluginPath = os.path.split(os.path.split(os.path.dirname(__file__))[0])[0]


class Hotspot(QgisAlgorithm):

    INPUT = 'INPUT_LAYER'
    FIELD = 'FIELD'
    NEIGHBORS = 'NEIGHBORS'
    OUTPUT = 'OUTPUT'

    def icon(self):
        # reuse clustering icon
        return QIcon(os.path.join(pluginPath, 'spatial_analysis', 'icons', 'cluster.svg'))

    def group(self):
        return self.tr('Hot Spot Analysis')

    def groupId(self):
        return 'hotspotanalysis'

    def name(self):
        return 'hotspot'

    def displayName(self):
        return self.tr('Hot Spot Analysis (Gi*)')

    def __init__(self):
        super().__init__()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.INPUT,
            self.tr('Input Layer'),
            [QgsProcessing.TypeVectorPolygon, QgsProcessing.TypeVectorPoint]))
        self.addParameter(QgsProcessingParameterField(
            self.FIELD,
            self.tr('Numeric Field'),
            parentLayerParameterName=self.INPUT,
            type=QgsProcessingParameterField.Numeric))
        self.addParameter(QgsProcessingParameterNumber(
            self.NEIGHBORS,
            self.tr('Number of Neighbors (k)'),
            QgsProcessingParameterNumber.Integer,
            8, False, 1, 50))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT,
            self.tr('Output Layer'),
            QgsProcessing.TypeVector))

    def processAlgorithm(self, parameters, context, feedback):
        if KNN is None or G_Local is None:
            help_file = os.path.join(
                pluginPath,
                'spatial_analysis',
                'help',
                'pysal_osgeo4w.html')
            url = QUrl.fromLocalFile(help_file).toString()
            msg = self.tr(
                'pysal 모듈이 필요합니다. '
                '<a href="{0}">OSGeo4W Shell에서 설치하는 방법</a>'.format(url))
            feedback.pushInfo(msg)
            raise QgsProcessingException(msg)

        feedback.pushInfo(self.tr("Starting Algorithm: '{}'".format(self.displayName())))
        layer = self.parameterAsSource(parameters, self.INPUT, context)
        field_name = self.parameterAsString(parameters, self.FIELD, context)
        k = self.parameterAsInt(parameters, self.NEIGHBORS, context)

        # Convert features to coordinate array and collect target values
        coords = []
        values = []
        for f in layer.getFeatures():
            geom = f.geometry().centroid()
            coords.append([geom.asPoint().x(), geom.asPoint().y()])
            values.append(f[field_name])

        if not coords:
            raise QgsProcessingException(self.tr('No features found.'))

        w = KNN.from_array(coords, k=k)
        g = G_Local(values, w)

        fields = layer.fields()
        new_fields = QgsFields()
        new_fields.append(QgsField('GiZScore', QVariant.Double))
        fields = QgsProcessingUtils.combineFields(fields, new_fields)
        (sink, dest_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            fields,
            layer.wkbType(),
            layer.sourceCrs())

        total = len(coords)
        for i, (feat, z) in enumerate(zip(layer.getFeatures(), g.Zs)):
            out_feat = feat
            attrs = feat.attributes()
            attrs.extend([float(z)])
            out_feat.setAttributes(attrs)
            sink.addFeature(out_feat, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(i / total * 100))
        feedback.setProgress(0)
        feedback.pushInfo(self.tr('Done with Hot Spot Layer'))

        results = {}
        results[self.OUTPUT] = dest_id
        return results
