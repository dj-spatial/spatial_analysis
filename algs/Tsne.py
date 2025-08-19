# -*- coding: utf-8 -*-
"""
/*******************************************************************************
                          A QGIS plugin
SpatialAnalyzer
                               -------------------
        git sha              : $Format:%H$
        copyright            : (C) 2024 by OpenAI
        email                : dev@example.com
*******************************************************************************/

/*******************************************************************************
 *                                                                             *
 *   This program is free software; you can redistribute it and/or modify      *
 *   it under the terms of the GNU General Public License as published by      *
 *   the Free Software Foundation; either version 2 of the License, or         *
 *   (at your option) any later version.                                       *
 *                                                                             *
 ******************************************************************************/
"""

__author__ = 'OpenAI'
__date__ = 'October 2024'
__copyright__ = '(C) 2024, OpenAI'

# This will get replaced with a git SHA1 when you do a git archive
__revision__ = '$Format:%H$'

import os
import numpy as np

from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtGui import QIcon

from qgis.core import (
    QgsField,
    QgsFields,
    QgsFeature,
    QgsFeatureSink,
    QgsProcessing,
    QgsProcessingException,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterNumber
)

from processing.algs.qgis.QgisAlgorithm import QgisAlgorithm
from spatial_analysis.forms.VariableParam import ParameterVariable

pluginPath = os.path.split(os.path.split(os.path.dirname(__file__))[0])[0]


class Tsne(QgisAlgorithm):
    """t-SNE algorithm referencing PySAL and scikit-learn."""

    INPUT = 'INPUT_LAYER'
    V_OPTIONS = 'V_OPTIONS'
    PERPLEXITY = 'PERPLEXITY'
    OUTPUT = 'OUTPUT'

    def icon(self):
        return QIcon(os.path.join(pluginPath, 'spatial_analysis', 'icons', 'browser.svg'))

    def group(self):
        return self.tr('Dimension Reduction')

    def groupId(self):
        return 'dimensionreduction'

    def name(self):
        return 'tsne'

    def displayName(self):
        return self.tr('t-SNE')

    def __init__(self):
        super().__init__()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(self.INPUT,
                                                              self.tr('Input Layer'),
                                                              [QgsProcessing.TypeVector]))
        variable_param = ParameterVariable(self.V_OPTIONS, self.tr('Variable Fields'), layer_param=self.INPUT)
        variable_param.setMetadata({'widget_wrapper': {'class': 'spatial_analysis.forms.VariableWidget.VariableWidgetWrapper'}})
        self.addParameter(variable_param)
        self.addParameter(QgsProcessingParameterNumber(self.PERPLEXITY,
                                                       self.tr('Perplexity'),
                                                       QgsProcessingParameterNumber.Double,
                                                       30.0, False, 5.0, 100.0))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT,
                                                            self.tr('Output Layer'),
                                                            QgsProcessing.TypeVector))

    def processAlgorithm(self, parameters, context, feedback):
        try:  # reference pysal and scikit-learn
            import libpysal  # noqa: F401
            from sklearn.manifold import TSNE
        except Exception as e:  # pragma: no cover - library may be missing
            raise QgsProcessingException(self.tr('Required libraries not found: {}').format(e))

        layer = self.parameterAsSource(parameters, self.INPUT, context)
        fields = self.parameterAsFields(parameters, self.V_OPTIONS, context)
        perplexity = self.parameterAsDouble(parameters, self.PERPLEXITY, context)

        data = []
        feats = list(layer.getFeatures())
        for feat in feats:
            row = [feat[field] for field in fields]
            data.append(row)
        data = np.array(data, dtype=float)

        tsne = TSNE(n_components=2, perplexity=perplexity, init='random', random_state=0)
        transformed = tsne.fit_transform(data)

        new_fields = QgsFields(layer.fields())
        new_fields.append(QgsField('TSNE1', QVariant.Double))
        new_fields.append(QgsField('TSNE2', QVariant.Double))

        (sink, dest_id) = self.parameterAsSink(parameters, self.OUTPUT, context,
                                               new_fields, layer.wkbType(), layer.sourceCrs())
        for feat, comp in zip(feats, transformed):
            new_feat = QgsFeature(feat)
            attrs = feat.attributes()
            attrs.extend(comp.tolist())
            new_feat.setAttributes(attrs)
            sink.addFeature(new_feat, QgsFeatureSink.FastInsert)

        return {self.OUTPUT: dest_id}
