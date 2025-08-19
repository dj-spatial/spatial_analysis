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
import codecs
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
    QgsProcessingParameterFileDestination,
    QgsCategorizedSymbolRenderer,
    QgsRendererCategory,
    QgsSymbol
)

from processing.algs.qgis.QgisAlgorithm import QgisAlgorithm

# pysal modules - these may not be available in all environments
try:
    from esda.getisord import G_Local
    from libpysal import weights as lp_weights
except Exception as e:  # pragma: no cover - library might be missing
    G_Local = None
    lp_weights = None

pluginPath = os.path.split(os.path.split(os.path.dirname(__file__))[0])[0]


class Hotspot(QgisAlgorithm):

    INPUT = 'INPUT_LAYER'
    FIELD = 'FIELD'
    WEIGHTS_BTN = 'WEIGHTS_BTN'
    OUTPUT = 'OUTPUT'
    WEIGHT_REPORT = 'WEIGHT_REPORT'

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
        weights_param = QgsProcessingParameterString(
            self.WEIGHTS_BTN,
            self.tr('Spatial Autocorrelation'),
            '', True)
        weights_param.setMetadata({'widget_wrapper': {
            'class': 'spatial_analysis.forms.WeightsWidget.WeightsWidgetWrapper',
            'layer_param': self.INPUT}})
        self.addParameter(weights_param)
        self.addParameter(QgsProcessingParameterField(
            self.FIELD,
            self.tr('Variable Field'),
            parentLayerParameterName=self.INPUT,
            type=QgsProcessingParameterField.Numeric))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT,
            self.tr('Output Layer'),
            QgsProcessing.TypeVector))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.WEIGHT_REPORT,
            self.tr('Weight Report'),
            'HTML files (*.html)'))

    def checkParameterValues(self, parameters, context):
        ok, msg = super().checkParameterValues(parameters, context)
        if not ok:
            return ok, msg
        if not parameters.get(self.WEIGHTS_BTN):
            return False, self.tr('Weights must be defined.')
        return True, ''

    def processAlgorithm(self, parameters, context, feedback):
        if G_Local is None:
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
        id_field = weight_info['id_field']

        # Ensure weights matrix uses floating point values to avoid
        # numba dtype issues within esda's G_Local implementation
        if lp_weights is not None and getattr(w.sparse, 'dtype', None) != np.float64:
            w = lp_weights.WSP(w.sparse.astype('float64'), ids=w.ids).to_W()

        id_to_feat = {}
        id_to_val = {}
        for f in layer.getFeatures():
            fid = f[id_field]
            id_to_feat[fid] = f
            id_to_val[fid] = f[field_name]

        try:
            values = np.asarray([id_to_val[i] for i in w.id_order], dtype=np.float64)
        except KeyError:
            raise QgsProcessingException(self.tr('ID field mismatch between weights and layer.'))

        if not np.isfinite(values).all():
            raise QgsProcessingException(self.tr('Selected field contains missing or infinite values.'))

         # 가중치 행렬도 float64 보장(안전망)
        if lp_weights is not None and getattr(w.sparse, 'dtype', None) != np.float64:
            w = lp_weights.WSP(w.sparse.astype(np.float64), ids=w.ids).to_W()
 
        g = G_Local(values, w, n_jobs=1)

        fields = layer.fields()
        new_fields = QgsFields()
        new_fields.append(QgsField('GiZScore', QVariant.Double))
        new_fields.append(QgsField('GiPValue', QVariant.Double))
        new_fields.append(QgsField('GiCluster', QVariant.Int))
        new_fields.append(QgsField('GiSig', QVariant.String, len=15))
        fields = QgsProcessingUtils.combineFields(fields, new_fields)
        (sink, dest_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            fields,
            layer.wkbType(),
            layer.sourceCrs())

        total = len(w.id_order)
        for i, (fid, z, p) in enumerate(zip(w.id_order, g.Zs, g.p_sim)):
            feat = id_to_feat[fid]
            attrs = feat.attributes()
            if p < 0.05:
                cluster = 1 if z > 0 else -1
            else:
                cluster = 0
            if cluster == 1:
                cat = 'High'
            elif cluster == -1:
                cat = 'Low'
            else:
                cat = 'Not Significant'
            attrs.extend([float(z), float(p), cluster, cat])
            feat.setAttributes(attrs)
            sink.addFeature(feat, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(i / total * 100))
        feedback.setProgress(0)
        feedback.pushInfo(self.tr('Done with Hot Spot Layer'))

        # Apply style to result layer
        result_layer = QgsProcessingUtils.mapLayerFromString(dest_id, context)
        category_colors = {
            'High': '#e31a1c',
            'Low': '#1f78b4',
            'Not Significant': '#d3d3d3'
        }
        categories = []
        for val, color in category_colors.items():
            symbol = QgsSymbol.defaultSymbol(result_layer.geometryType())
            symbol.setColor(QColor(color))
            category = QgsRendererCategory(val, symbol, val)
            categories.append(category)
        renderer = QgsCategorizedSymbolRenderer('GiSig', categories)
        result_layer.setRenderer(renderer)
        result_layer.triggerRepaint()
        result_layer.setName('Gi*')
        style_path = os.path.join(QgsProcessingUtils.tempFolder(), 'hotspot_gistar.qml')
        result_layer.saveNamedStyle(style_path)
        if context.willLoadLayerOnCompletion(dest_id):
            details = context.layersToLoadOnCompletion()[dest_id]
            details.name = 'Gi*'      # 완료 후 자동 로드될 때 표시 이름 지정
            details.style = style_path

        weight_report = self.parameterAsFileOutput(parameters, self.WEIGHT_REPORT, context)
        summary = weight_info.get('summary', '')
        with codecs.open(weight_report, 'w', encoding='utf-8') as f:
            f.write('<html><head>\n')
            f.write('<meta http-equiv="Content-Type" content="text/html; charset=utf-8" /></head><body>\n')
            f.write('<pre>{0}</pre>\n'.format(summary))
            f.write('</body></html>')
        return {self.OUTPUT: dest_id}
