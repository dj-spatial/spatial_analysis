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
import inspect
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


class Tsne(QgisAlgorithm):
    """t-SNE algorithm referencing PySAL and scikit-learn."""

    INPUT = 'INPUT_LAYER'
    FIELDS = 'FIELDS'
    PERPLEXITY = 'PERPLEXITY'
    THETA = 'THETA'
    MAX_ITER = 'MAX_ITER'
    LEARNING_RATE = 'LEARNING_RATE'
    MOMENTUM = 'MOMENTUM'
    FINAL_MOMENTUM = 'FINAL_MOMENTUM'
    SWITCH_ITER = 'SWITCH_ITER'
    DISTANCE = 'DISTANCE'
    CATEGORY = 'CATEGORY'
    TRANSFORMATION = 'TRANSFORMATION'
    USE_SEED = 'USE_SEED'
    SEED = 'SEED'
    OUTPUT = 'OUTPUT'
    REPORT = 'REPORT'

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
        self.addParameter(QgsProcessingParameterNumber(self.PERPLEXITY,
                                                       self.tr('Perplexity'),
                                                       QgsProcessingParameterNumber.Double,
                                                       30.0, False, 5.0, 100.0))
        self.addParameter(QgsProcessingParameterNumber(self.THETA,
                                                       self.tr('Theta'),
                                                       QgsProcessingParameterNumber.Double,
                                                       0.5, False, 0.0, 1.0))
        self.addParameter(QgsProcessingParameterNumber(self.MAX_ITER,
                                                       self.tr('Max Iteration'),
                                                       QgsProcessingParameterNumber.Integer,
                                                       5000, False, 250, 10000))
        self.addParameter(QgsProcessingParameterNumber(self.LEARNING_RATE,
                                                       self.tr('Learning Rate'),
                                                       QgsProcessingParameterNumber.Double,
                                                       200.0, False, 10.0, 1000.0))
        self.addParameter(QgsProcessingParameterNumber(self.MOMENTUM,
                                                       self.tr('Momentum'),
                                                       QgsProcessingParameterNumber.Double,
                                                       0.5, False, 0.0, 1.0))
        self.addParameter(QgsProcessingParameterNumber(self.FINAL_MOMENTUM,
                                                       self.tr('Final Momentum'),
                                                       QgsProcessingParameterNumber.Double,
                                                       0.8, False, 0.0, 1.0))
        self.addParameter(QgsProcessingParameterNumber(self.SWITCH_ITER,
                                                       self.tr('# Iteration Switch Momentum'),
                                                       QgsProcessingParameterNumber.Integer,
                                                       250, False, 1, 10000))
        self.addParameter(QgsProcessingParameterEnum(
            self.DISTANCE,
            self.tr('Distance Function'),
            ['Euclidean', 'Manhattan'],
            defaultValue=0
        ))
        self.addParameter(QgsProcessingParameterField(self.CATEGORY,
                                                      self.tr('Category Variable'),
                                                      parentLayerParameterName=self.INPUT,
                                                      optional=True))
        self.addParameter(QgsProcessingParameterEnum(
            self.TRANSFORMATION,
            self.tr('Method'),
            ['Standardize (Z)', 'Standardize MAD', 'Range Adjust', 'Range Standardize', 'Raw', 'Demean'],
            defaultValue=0
        ))
        self.addParameter(QgsProcessingParameterBoolean(self.USE_SEED,
                                                        self.tr('Use Specified Seed'),
                                                        False))
        self.addParameter(QgsProcessingParameterNumber(self.SEED,
                                                       self.tr('Seed'),
                                                       QgsProcessingParameterNumber.Integer,
                                                       0, True))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT,
                                                            self.tr('Output Layer'),
                                                            QgsProcessing.TypeVector))
        self.addOutput(QgsProcessingOutputHtml(self.REPORT, self.tr('Scatter Plot')))


    def processAlgorithm(self, parameters, context, feedback):
        try:  # reference pysal and t-SNE backend
            import libpysal  # noqa: F401
            try:
                from openTSNE import TSNE  # prefer openTSNE if available
            except Exception:
                from sklearn.manifold import TSNE  # fall back to scikit-learn
        except Exception as e:  # pragma: no cover - library may be missing
            raise QgsProcessingException(self.tr('Required libraries not found: {}').format(e))

        layer = self.parameterAsSource(parameters, self.INPUT, context)
        fields = self.parameterAsFields(parameters, self.FIELDS, context)
        if not fields:
            raise QgsProcessingException(self.tr('No Fields Selected.'))
        perplexity = self.parameterAsDouble(parameters, self.PERPLEXITY, context)
        theta = self.parameterAsDouble(parameters, self.THETA, context)
        max_iter = self.parameterAsInt(parameters, self.MAX_ITER, context)
        learning_rate = self.parameterAsDouble(parameters, self.LEARNING_RATE, context)
        momentum = self.parameterAsDouble(parameters, self.MOMENTUM, context)
        final_momentum = self.parameterAsDouble(parameters, self.FINAL_MOMENTUM, context)
        switch_iter = self.parameterAsInt(parameters, self.SWITCH_ITER, context)
        metric = ['euclidean', 'manhattan'][self.parameterAsEnum(parameters, self.DISTANCE, context)]
        category = self.parameterAsString(parameters, self.CATEGORY, context)
        transform = ['standardize', 'standardize_mad', 'range_adjust', 'range_standardize', 'raw', 'demean'][
            self.parameterAsEnum(parameters, self.TRANSFORMATION, context)]
        use_seed = self.parameterAsBoolean(parameters, self.USE_SEED, context)
        seed = self.parameterAsInt(parameters, self.SEED, context) if use_seed else None

        feats = list(layer.getFeatures())
        data = [[f[fld] for fld in fields] for f in feats]
        data = np.array(data, dtype=float)
        if transform == 'standardize':
            data_proc = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        elif transform == 'standardize_mad':
            med = np.median(data, axis=0)
            mad = np.median(np.abs(data - med), axis=0)
            data_proc = (data - med) / mad
        elif transform == 'range_adjust':
            minv = np.min(data, axis=0)
            maxv = np.max(data, axis=0)
            data_proc = (data - minv) / (maxv - minv)
        elif transform == 'range_standardize':
            minv = np.min(data, axis=0)
            maxv = np.max(data, axis=0)
            rangev = maxv - minv
            meanv = (maxv + minv) / 2
            data_proc = (data - meanv) / rangev
        elif transform == 'demean':
            data_proc = data - np.mean(data, axis=0)
        else:
            data_proc = data

        log_interval = 50

        def _callback(iteration, error, embedding):
            feedback.pushInfo(f'Iteration {iteration}: error is {error}')
            if iteration % (log_interval * 10) == 0 or iteration == max_iter:
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.scatter(embedding[:, 0], embedding[:, 1], s=10)
                ax.set_title(f'Iteration {iteration}')
                img_path = os.path.join(tempfile.gettempdir(), f'tsne_{iteration}.png')
                fig.savefig(img_path)
                plt.close(fig)

        params = {
            'n_components': 2,
            'perplexity': perplexity,
            'learning_rate': learning_rate,
            'metric': metric,
            'random_state': seed,
            'verbose': 0
        }

        sig = inspect.signature(TSNE.__init__)
        if 'angle' in sig.parameters:
            params['angle'] = theta
        if 'init' in sig.parameters:
            params['init'] = 'random'
        if 'n_iter' in sig.parameters:
            params['n_iter'] = max_iter
        elif 'max_iter' in sig.parameters:
            params['max_iter'] = max_iter
        if 'callbacks' in sig.parameters and 'callbacks_every_iters' in sig.parameters:
            params['callbacks'] = _callback
            params['callbacks_every_iters'] = log_interval
            tsne = TSNE(**params)
            transformed = tsne.fit(data_proc)
            final_cost = getattr(tsne, 'kl_divergence_', float('nan'))
        else:
            tsne = TSNE(**params)
            transformed = tsne.fit_transform(data_proc)
            final_cost = getattr(tsne, 'kl_divergence_', float('nan'))
        feedback.pushInfo(self.tr('final cost: {}').format(final_cost))

        new_fields = QgsFields(layer.fields())
        new_fields.append(QgsField('TSNE1', QVariant.Double))
        new_fields.append(QgsField('TSNE2', QVariant.Double))

        if category:
            new_fields.append(QgsField('CATEGORY', QVariant.String))

        (sink, dest_id) = self.parameterAsSink(parameters, self.OUTPUT, context,
                                               new_fields, layer.wkbType(), layer.sourceCrs())
        for feat, comp in zip(feats, transformed):
            new_feat = QgsFeature(feat)
            attrs = feat.attributes()
            attrs.extend(comp.tolist())
            if category:
                attrs.append(feat[category])
            new_feat.setAttributes(attrs)
            sink.addFeature(new_feat, QgsFeatureSink.FastInsert)
        fig, ax = plt.subplots(figsize=(12, 8))
        if category:
            cats = [f[category] for f in feats]
            unique = list(dict.fromkeys(cats))
            for cat_val in unique:
                idx = [i for i, c in enumerate(cats) if c == cat_val]
                ax.scatter(transformed[idx, 0], transformed[idx, 1], s=10, label=str(cat_val))
            ax.legend()
        else:
            ax.scatter(transformed[:, 0], transformed[:, 1], s=10)
        ax.set_xlabel('TSNE1')
        ax.set_ylabel('TSNE2')
        ax.set_title('t-SNE Scatter Plot')
        plot_path = os.path.join(tempfile.gettempdir(), 'tsne_scatter_plot.png')
        fig.savefig(plot_path)
        plt.close(fig)

        html = [
            '<html><head><meta charset="utf-8"/></head><body>',
            f'<img src="{plot_path}" alt="t-SNE Scatter Plot" style="width:100%;height:auto;"/>',
            '</body></html>'
        ]
        report_path = os.path.join(tempfile.gettempdir(), 'tsne_report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html))

        return {self.OUTPUT: dest_id, self.REPORT: report_path}