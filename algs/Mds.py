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
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

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
    QgsProcessingParameterNumber,
    QgsProcessingParameterEnum,
    QgsProcessingParameterField,
    QgsProcessingParameterBoolean,
    QgsProcessingOutputHtml
)

from processing.algs.qgis.QgisAlgorithm import QgisAlgorithm

pluginPath = os.path.split(os.path.split(os.path.dirname(__file__))[0])[0]


class Mds(QgisAlgorithm):
    """Multidimensional scaling inspired by GeoDa and PySAL implementations."""

    INPUT = 'INPUT_LAYER'
    FIELDS = 'FIELDS'
    DIMENSIONS = 'DIMENSIONS'
    TRANSFORMATION = 'TRANSFORMATION'
    METRIC = 'METRIC'
    DISTANCE = 'DISTANCE'
    MAX_ITER = 'MAX_ITER'
    EPS = 'EPS'
    N_INIT = 'N_INIT'
    USE_SEED = 'USE_SEED'
    SEED = 'SEED'
    CATEGORY = 'CATEGORY'
    OUTPUT = 'OUTPUT'
    REPORT = 'REPORT'

    def icon(self):
        return QIcon(os.path.join(pluginPath, 'spatial_analysis', 'icons', 'dimension.svg'))

    def group(self):
        return self.tr('Dimension Reduction')

    def groupId(self):
        return 'dimensionreduction'

    def name(self):
        return 'mds'

    def displayName(self):
        return self.tr('Multidimensional Scaling (MDS)')

    def __init__(self):
        super().__init__()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(self.INPUT,
                                                              self.tr('Input Layer'),
                                                              [QgsProcessing.TypeVector]))
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELDS,
                self.tr('Variable Fields'),
                None,
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric,
                allowMultiple=True
            )
        )
        self.addParameter(QgsProcessingParameterNumber(self.DIMENSIONS,
                                                       self.tr('Number of Dimensions'),
                                                       QgsProcessingParameterNumber.Integer,
                                                       2, False, 1, 6))
        self.addParameter(QgsProcessingParameterEnum(
            self.TRANSFORMATION,
            self.tr('Method'),
            ['Standardize (Z)', 'Standardize MAD', 'Range Adjust', 'Range Standardize', 'Raw', 'Demean'],
            defaultValue=0
        ))
        self.addParameter(QgsProcessingParameterEnum(
            self.METRIC,
            self.tr('Scaling Type'),
            ['Metric', 'Non-metric'],
            defaultValue=0
        ))
        self.addParameter(QgsProcessingParameterEnum(
            self.DISTANCE,
            self.tr('Distance Function'),
            ['Euclidean', 'Manhattan', 'Chebyshev'],
            defaultValue=0
        ))
        self.addParameter(QgsProcessingParameterNumber(self.MAX_ITER,
                                                       self.tr('Maximum Iterations'),
                                                       QgsProcessingParameterNumber.Integer,
                                                       300, False, 50, 1000))
        self.addParameter(QgsProcessingParameterNumber(self.EPS,
                                                       self.tr('Convergence Tolerance (eps)'),
                                                       QgsProcessingParameterNumber.Double,
                                                       1e-3, False, 1e-6, 1e-1))
        self.addParameter(QgsProcessingParameterNumber(self.N_INIT,
                                                       self.tr('Number of Initializations'),
                                                       QgsProcessingParameterNumber.Integer,
                                                       4, False, 1, 10))
        self.addParameter(QgsProcessingParameterBoolean(self.USE_SEED,
                                                        self.tr('Use Specified Seed'),
                                                        False))
        self.addParameter(QgsProcessingParameterNumber(self.SEED,
                                                       self.tr('Seed'),
                                                       QgsProcessingParameterNumber.Integer,
                                                       0, True))
        self.addParameter(QgsProcessingParameterField(self.CATEGORY,
                                                      self.tr('Category Variable (optional)'),
                                                      parentLayerParameterName=self.INPUT,
                                                      optional=True))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT,
                                                            self.tr('Output Layer'),
                                                            QgsProcessing.TypeVector))
        self.addOutput(QgsProcessingOutputHtml(self.REPORT, self.tr('Scatter Plot')))

    def _transform_data(self, data, transform):
        if transform == 'standardize':
            std = np.std(data, axis=0)
            std[std == 0] = 1
            return (data - np.mean(data, axis=0)) / std
        if transform == 'demean':
            return data - np.mean(data, axis=0)
        if transform == 'standardize_mad':
            med = np.median(data, axis=0)
            mad = np.median(np.abs(data - med), axis=0)
            mad[mad == 0] = 1
            return (data - med) / mad
        if transform == 'range_adjust':
            minv = np.min(data, axis=0)
            maxv = np.max(data, axis=0)
            span = maxv - minv
            span[span == 0] = 1
            return (data - minv) / span
        if transform == 'range_standardize':
            minv = np.min(data, axis=0)
            maxv = np.max(data, axis=0)
            rangev = maxv - minv
            rangev[rangev == 0] = 1
            meanv = (maxv + minv) / 2
            return (data - meanv) / rangev
        return data

    def _pairwise(self, data_proc, distance_choice):
        from sklearn.metrics import pairwise_distances

        metrics = {
            0: 'euclidean',
            1: 'manhattan',
            2: 'chebyshev'
        }
        metric = metrics.get(distance_choice, 'euclidean')
        return pairwise_distances(data_proc, metric=metric)

    def processAlgorithm(self, parameters, context, feedback):
        try:
            import libpysal  # noqa: F401
        except Exception as e:  # pragma: no cover - library may be missing
            raise QgsProcessingException(self.tr('PySAL is required: {}').format(e))

        try:
            from sklearn.manifold import MDS as SKMDS
        except Exception as e:  # pragma: no cover - library may be missing
            raise QgsProcessingException(self.tr('scikit-learn is required for MDS: {}').format(e))

        layer = self.parameterAsSource(parameters, self.INPUT, context)
        fields = self.parameterAsFields(parameters, self.FIELDS, context)
        if not fields:
            raise QgsProcessingException(self.tr('No Fields Selected.'))

        n_dim = self.parameterAsInt(parameters, self.DIMENSIONS, context)
        if n_dim < 1:
            raise QgsProcessingException(self.tr('Dimensions must be at least 1.'))
        if n_dim > len(fields):
            feedback.pushInfo(
                self.tr('Reducing dimensions from {0} to {1} to match selected variables').format(n_dim, len(fields))
            )
            n_dim = len(fields)

        transform_labels = ['Standardize (Z)', 'Standardize MAD', 'Range Adjust', 'Range Standardize', 'Raw', 'Demean']
        transform_keys = ['standardize', 'standardize_mad', 'range_adjust', 'range_standardize', 'raw', 'demean']
        transform_idx = self.parameterAsEnum(parameters, self.TRANSFORMATION, context)
        transform_opt = transform_keys[transform_idx]
        transform_label = transform_labels[transform_idx]
        metric = self.parameterAsEnum(parameters, self.METRIC, context) == 0
        distance_choice = self.parameterAsEnum(parameters, self.DISTANCE, context)
        max_iter = self.parameterAsInt(parameters, self.MAX_ITER, context)
        eps = self.parameterAsDouble(parameters, self.EPS, context)
        n_init = self.parameterAsInt(parameters, self.N_INIT, context)
        use_seed = self.parameterAsBool(parameters, self.USE_SEED, context)
        seed = self.parameterAsInt(parameters, self.SEED, context)
        cat_field_name = self.parameterAsString(parameters, self.CATEGORY, context)
        cat_field = cat_field_name if cat_field_name else None

        feats = list(layer.getFeatures())
        if not feats:
            raise QgsProcessingException(self.tr('Input layer has no features.'))

        data = []
        categories = []
        for feat in feats:
            row = []
            for fld in fields:
                val = feat[fld]
                if val is None:
                    raise QgsProcessingException(
                        self.tr('Feature {0} has NULL value in field {1}.').format(feat.id(), fld)
                    )
                row.append(float(val))
            data.append(row)
            if cat_field is not None:
                categories.append(feat[cat_field])
            else:
                categories.append(None)

        data = np.array(data, dtype=float)
        data_proc = self._transform_data(data, transform_opt)

        dissimilarities = self._pairwise(data_proc, distance_choice)

        random_state = seed if use_seed else None
        model = SKMDS(
            n_components=n_dim,
            metric=metric,
            max_iter=max_iter,
            eps=eps,
            n_init=n_init,
            dissimilarity='precomputed',
            random_state=random_state,
        )

        embedding = model.fit_transform(dissimilarities)
        stress = getattr(model, 'stress_', np.nan)
        n_iter = getattr(model, 'n_iter_', np.nan)

        new_fields = QgsFields(layer.fields())
        for i in range(n_dim):
            new_fields.append(QgsField(f'MDS{i+1}', QVariant.Double))

        (sink, dest_id) = self.parameterAsSink(parameters, self.OUTPUT, context,
                                               new_fields, layer.wkbType(), layer.sourceCrs())

        for feat, coords in zip(feats, embedding):
            new_feat = QgsFeature(feat)
            attrs = feat.attributes()
            attrs.extend(coords.tolist())
            new_feat.setAttributes(attrs)
            sink.addFeature(new_feat, QgsFeatureSink.FastInsert)

        scatter_path = None
        if n_dim >= 2:
            fig, ax = plt.subplots()
            x = embedding[:, 0]
            y = embedding[:, 1]
            if cat_field is not None:
                cats = np.array(categories)
                unique = np.unique(cats)
                for cat in unique:
                    mask = cats == cat
                    ax.scatter(x[mask], y[mask], label=str(cat))
                ax.legend(loc='best', fontsize='small')
            else:
                ax.scatter(x, y)
            ax.set_xlabel('MDS1')
            ax.set_ylabel('MDS2')
            ax.set_title('MDS Embedding')
            scatter_path = os.path.join(tempfile.gettempdir(), 'mds_scatter.png')
            fig.savefig(scatter_path)
            plt.close(fig)

        distance_labels = [self.tr('Euclidean'), self.tr('Manhattan'), self.tr('Chebyshev')]
        html_parts = [
            '<html><head><meta charset="utf-8"/></head><body>',
            f'<p>{self.tr("Scaling type")}: {"Metric" if metric else "Non-metric"}</p>',
            f'<p>{self.tr("Distance function")}: {distance_labels[distance_choice]}</p>',
            f'<p>{self.tr("Stress")}: {stress:.6f}</p>',
            f'<p>{self.tr("Iterations")}: {n_iter}</p>',
            f'<p>{self.tr("Transformation")}: {transform_label}</p>',
        ]

        coord_header = ' '.join([f'MDS{i+1:>2}' for i in range(n_dim)])
        coord_lines = [f'{feat.id():<10}' + ' '.join(f'{v:10.6f}' for v in coords) for feat, coords in zip(feats, embedding)]
        html_parts.append('<p>MDS coordinates:<br><pre>' + ' ' * 10 + coord_header + '\n' + '\n'.join(coord_lines) + '</pre></p>')
        if scatter_path:
            html_parts.append(f'<img src="{scatter_path}" alt="MDS Scatter"/>')
        html_parts.append('</body></html>')

        report_path = os.path.join(tempfile.gettempdir(), 'mds_report.html')
        with open(report_path, 'w', encoding='utf-8') as handle:
            handle.write('\n'.join(html_parts))

        feedback.pushInfo(self.tr('Stress:') + f' {stress:.6f}')
        feedback.pushInfo(self.tr('Iterations:') + f' {n_iter}')

        return {self.OUTPUT: dest_id, self.REPORT: report_path}
