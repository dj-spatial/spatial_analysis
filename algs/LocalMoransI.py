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

import numpy as np

from qgis.PyQt.QtCore import QVariant, QUrl
from qgis.PyQt.QtGui import QIcon, QColor

from qgis.core import (
    QgsField,
    QgsFields,
    QgsProcessing,
    QgsProcessingException,
    QgsProcessingUtils,
    QgsFeatureSink,
    QgsProcessingParameterField,
    QgsProcessingParameterString,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFeatureSink,
    QgsVectorLayer,
    QgsCategorizedSymbolRenderer,
    QgsRendererCategory,
    QgsSymbol,
    QgsWkbTypes,
    QgsProcessingLayerPostProcessorInterface
)

from processing.algs.qgis.QgisAlgorithm import QgisAlgorithm

# pysal modules - these may not be available in all environments
try:
    from esda.moran import Moran_Local
except Exception as e:  # pragma: no cover - library might be missing
    Moran_Local = None

pluginPath = os.path.split(os.path.split(os.path.dirname(__file__))[0])[0]


class LocalMoransI(QgisAlgorithm):

    INPUT = 'INPUT_LAYER'
    FIELD = 'FIELD'
    WEIGHTS_BTN = 'WEIGHTS_BTN'
    OUTPUT = 'OUTPUT'

    def icon(self):
        # reuse clustering icon
        return QIcon(os.path.join(pluginPath, 'spatial_analysis', 'icons', 'cluster.svg'))

    def group(self):
        return self.tr('Hot Spot Analysis')

    def groupId(self):
        return 'hotspotanalysis'

    def name(self):
        return 'localmoransi'

    def displayName(self):
        return self.tr("Local Moran's i")

    def __init__(self):
        super().__init__()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.INPUT,
            self.tr('Input Layer'),
            [QgsProcessing.TypeVectorPolygon, QgsProcessing.TypeVectorPoint]))
        weights_param = QgsProcessingParameterString(
            self.WEIGHTS_BTN,
            self.tr('Weights'),
            '', True)
        weights_param.setMetadata({'widget_wrapper': {
            'class': 'spatial_analysis.forms.WeightsWidget.WeightsWidgetWrapper',
            'layer_param': self.INPUT}})
        self.addParameter(weights_param)
        self.addParameter(QgsProcessingParameterField(
            self.FIELD,
            self.tr('Numeric Field'),
            parentLayerParameterName=self.INPUT,
            type=QgsProcessingParameterField.Numeric))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT,
            self.tr('Output Layer'),
            QgsProcessing.TypeVector))

    def processAlgorithm(self, parameters, context, feedback):
        if Moran_Local is None:
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
        weight_info = parameters.get(self.WEIGHTS_BTN)
        if not weight_info:
            raise QgsProcessingException(self.tr('Weights must be defined.'))

        w = weight_info['weights']
        if getattr(w, 'transform', '') != 'R':
            w.transform = 'R'
        id_field = weight_info['id_field']

        id_to_feat = {}
        id_to_val = {}
        for f in layer.getFeatures():
            fid = f[id_field]
            id_to_feat[fid] = f
            id_to_val[fid] = f[field_name]

        try:
            values = [id_to_val[i] for i in w.id_order]
        except KeyError:
            raise QgsProcessingException(self.tr('ID field mismatch between weights and layer.'))

        m = Moran_Local(values, w, permutations=999)

        # Local Moran's I values
        local_i = getattr(m, 'Is', None)
        if local_i is None:
            # some versions expose Ii as .I or .Is
            local_i = getattr(m, 'I', [])

        # GeoDa reports analytical z-scores; compute them if available
        if hasattr(m, 'z'):
            local_z = m.z
        else:
            ei = getattr(m, 'EI', getattr(m, 'EI_sim', None))
            vi = getattr(m, 'VI_rand', getattr(m, 'VI_sim', None))
            if ei is not None and vi is not None:
                ei = np.asarray(ei)
                vi = np.asarray(vi)
                with np.errstate(invalid='ignore'):
                    local_z = (np.asarray(local_i) - ei) / np.sqrt(vi)
            else:
                local_z = m.z_sim

        fields = layer.fields()
        new_fields = QgsFields()
        new_fields.append(QgsField('LocalI', QVariant.Double))
        new_fields.append(QgsField('LocalIZ', QVariant.Double))
        new_fields.append(QgsField('PValue', QVariant.Double))
        new_fields.append(QgsField('Cluster', QVariant.String, len=10))
        fields = QgsProcessingUtils.combineFields(fields, new_fields)
        (sink, dest_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            fields,
            layer.wkbType(),
            layer.sourceCrs())

        cluster_map = {1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL'}
        clusters = []
        for q, p in zip(m.q, m.p_sim):
            if p < 0.05:
                clusters.append(cluster_map.get(q, 'NotSig'))
            else:
                clusters.append('NotSig')

        total = len(w.id_order)
        for i, (fid, Ii, z, p, c) in enumerate(zip(w.id_order, local_i, local_z, m.p_sim, clusters)):
            feat = id_to_feat[fid]
            attrs = feat.attributes()
            attrs.extend([float(Ii), float(z), float(p), c])
            feat.setAttributes(attrs)
            sink.addFeature(feat, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(i / total * 100))
        feedback.setProgress(0)
        feedback.pushInfo(self.tr('Done with Local Moran Layer'))

        # 결과 레이어 스타일 적용
        result_layer = QgsProcessingUtils.mapLayerFromString(dest_id, context)

        # Cluster 값에 따른 색상 정의
        category_colors = {
            'HH': '#e31a1c',     # High-High
            'HL': '#fb9a99',     # High-Low
            'LH': '#a6cee3',     # Low-High
            'LL': '#1f78b4',     # Low-Low
            'NotSig': '#d3d3d3'  # Not Significant
        }

        from qgis.core import QgsSymbol, QgsRendererCategory, QgsCategorizedSymbolRenderer
        from PyQt5.QtGui import QColor

        categories = []
        for val, color in category_colors.items():
            symbol = QgsSymbol.defaultSymbol(result_layer.geometryType())
            symbol.setColor(QColor(color))
            category = QgsRendererCategory(val, symbol, val)
            categories.append(category)

        renderer = QgsCategorizedSymbolRenderer('Cluster', categories)
        result_layer.setRenderer(renderer)
        result_layer.triggerRepaint()

        # QML 스타일 파일을 임시로 저장
        style_path = os.path.join(QgsProcessingUtils.tempFolder(), 'local_moran_cluster.qml')
        result_layer.saveNamedStyle(style_path)

        # context에 스타일 경로 설정 (자동 로드시 적용됨)
        if context.willLoadLayerOnCompletion(dest_id):
            context.layersToLoadOnCompletion()[dest_id].style = style_path

        results = {}
        results[self.OUTPUT] = dest_id
        return results
