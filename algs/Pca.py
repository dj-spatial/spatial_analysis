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


class Pca(QgisAlgorithm):
    """Simple PCA algorithm referencing PySAL."""

    INPUT = 'INPUT_LAYER'
    V_OPTIONS = 'V_OPTIONS'
    N_COMPONENTS = 'N_COMPONENTS'
    OUTPUT = 'OUTPUT'

    def icon(self):
        return QIcon(os.path.join(pluginPath, 'spatial_analysis', 'icons', 'browser.svg'))

    def group(self):
        return self.tr('Dimension Reduction')

    def groupId(self):
        return 'dimensionreduction'

    def name(self):
        return 'pca'

    def displayName(self):
        return self.tr('Principal Component Analysis')

    def __init__(self):
        super().__init__()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(self.INPUT,
                                                              self.tr('Input Layer'),
                                                              [QgsProcessing.TypeVector]))
        variable_param = ParameterVariable(self.V_OPTIONS, self.tr('Variable Fields'), layer_param=self.INPUT)
        variable_param.setMetadata({'widget_wrapper': {'class': 'spatial_analysis.forms.VariableWidget.VariableWidgetWrapper'}})
        self.addParameter(variable_param)
        self.addParameter(QgsProcessingParameterNumber(self.N_COMPONENTS,
                                                       self.tr('Number of Components'),
                                                       QgsProcessingParameterNumber.Integer,
                                                       2, False, 1, 10))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT,
                                                            self.tr('Output Layer'),
                                                            QgsProcessing.TypeVector))

    def processAlgorithm(self, parameters, context, feedback):
        try:  # reference pysal for compliance with request
            import libpysal  # noqa: F401
        except Exception as e:  # pragma: no cover - library may be missing
            raise QgsProcessingException(self.tr('PySAL is required: {}').format(e))

        layer = self.parameterAsSource(parameters, self.INPUT, context)
        fields = self.parameterAsFields(parameters, self.V_OPTIONS, context)
        n_comp = self.parameterAsInt(parameters, self.N_COMPONENTS, context)

        data = []
        feats = list(layer.getFeatures())
        for feat in feats:
            row = [feat[field] for field in fields]
            data.append(row)
        data = np.array(data, dtype=float)

        if data.shape[1] < n_comp:
            raise QgsProcessingException(self.tr('Number of components exceeds number of variables'))

        data -= np.mean(data, axis=0)
        cov = np.cov(data, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvecs = eigvecs[:, order][:, :n_comp]
        transformed = np.dot(data, eigvecs)

        new_fields = QgsFields(layer.fields())
        for i in range(n_comp):
            new_fields.append(QgsField(f'PC{i+1}', QVariant.Double))

        (sink, dest_id) = self.parameterAsSink(parameters, self.OUTPUT, context,
                                               new_fields, layer.wkbType(), layer.sourceCrs())
        for feat, comp in zip(feats, transformed):
            new_feat = QgsFeature(feat)
            attrs = feat.attributes()
            attrs.extend(comp.tolist())
            new_feat.setAttributes(attrs)
            sink.addFeature(new_feat, QgsFeatureSink.FastInsert)

        return {self.OUTPUT: dest_id}