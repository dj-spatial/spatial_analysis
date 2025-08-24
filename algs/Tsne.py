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
import inspect
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

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
    BACKEND = 'BACKEND'
    USE_SEED = 'USE_SEED'
    SEED = 'SEED'
    SHOW_EVOLUTION = 'SHOW_EVOLUTION'
    OUTPUT = 'OUTPUT'
    REPORT = 'REPORT'

    def icon(self):
        return QIcon(os.path.join(pluginPath, 'spatial_analysis', 'icons', 'dimension.svg'))

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
        backend_param = QgsProcessingParameterEnum(
            self.BACKEND,
            self.tr('Backend'),
            ['openTSNE', 'scikit-learn'],
            defaultValue=0
        )
        backend_param.setMetadata({'widget_wrapper': {'useRadioButtons': True}})
        self.addParameter(backend_param)
        self.addParameter(QgsProcessingParameterBoolean(self.USE_SEED,
                                                        self.tr('Use Specified Seed'),
                                                        False))
        self.addParameter(QgsProcessingParameterNumber(self.SEED,
                                                       self.tr('Seed'),
                                                       QgsProcessingParameterNumber.Integer,
                                                       0, True))
        self.addParameter(QgsProcessingParameterBoolean(self.SHOW_EVOLUTION,
                                                        self.tr('View iteration embedding process'),
                                                        False))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT,
                                                            self.tr('Output Layer'),
                                                            QgsProcessing.TypeVector))
        self.addOutput(QgsProcessingOutputHtml(self.REPORT, self.tr('Scatter Plot')))


    def processAlgorithm(self, parameters, context, feedback):
        try:
            import libpysal  # noqa: F401
        except Exception as e:  # pragma: no cover - library may be missing
            raise QgsProcessingException(
                self.tr('Required libraries not found or incomplete: {}').format(e)
            )

        backend_idx = self.parameterAsEnum(parameters, self.BACKEND, context)
        if backend_idx == 0:
            try:
                from openTSNE import TSNE as OTSNE
            except Exception as e:  # pragma: no cover - library may be missing
                raise QgsProcessingException(
                    self.tr('openTSNE library not found or incomplete: {}').format(e)
                )
        else:
            try:
                from sklearn.manifold import TSNE as SKTSNE
            except Exception as e:  # pragma: no cover - library may be missing
                raise QgsProcessingException(
                    self.tr('scikit-learn library not found or incomplete: {}').format(e)
                )
            sig = inspect.signature(SKTSNE.__init__)
            iter_kw = 'n_iter' if 'n_iter' in sig.parameters else 'max_iter'

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
        show_evolution = self.parameterAsBoolean(parameters, self.SHOW_EVOLUTION, context)

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

        if backend_idx == 0:  # openTSNE
            if show_evolution:
                iterations = []
                errors = []
                embeddings = []
                log_interval = max(1, max_iter // 200)

                def _callback(iteration, error, embedding):
                    iterations.append(iteration)
                    errors.append(error)
                    embeddings.append(embedding.copy())
                    return False

                params = {
                    'n_components': 2,
                    'perplexity': perplexity,
                    'learning_rate': learning_rate,
                    'metric': metric,
                    'theta': theta,
                    'n_iter': max_iter,
                    'random_state': seed,
                    'verbose': 0,
                    'callbacks': _callback,
                    'callbacks_every_iters': log_interval
                }
            else:
                iterations = errors = embeddings = None
                params = {
                    'n_components': 2,
                    'perplexity': perplexity,
                    'learning_rate': learning_rate,
                    'metric': metric,
                    'theta': theta,
                    'n_iter': max_iter,
                    'random_state': seed,
                    'verbose': 0
                }

            tsne = OTSNE(**params)
            transformed = np.asarray(tsne.fit(data_proc))
            final_cost = getattr(tsne, 'kl_divergence_', float('nan'))
            if show_evolution:
                frames = embeddings[:-1]
                frame_iters = iterations[:-1]
            else:
                frames = frame_iters = None

        else:  # scikit-learn
            if show_evolution:
                if max_iter < 250:
                    raise QgsProcessingException(self.tr('scikit-learn backend requires max_iter >= 250'))
                log_interval = max(250, max_iter // 200)
                iterations = []
                errors = []
                embeddings = []
                current_iter = 0
                current_init = 'pca'
                exaggeration = 12.0
                while current_iter < max_iter:
                    step = min(log_interval, max_iter - current_iter)
                    step = max(250, step)
                    params = {
                        'n_components': 2,
                        'perplexity': perplexity,
                        'learning_rate': learning_rate,
                        'metric': metric,
                        iter_kw: step,
                        'init': current_init,
                        'early_exaggeration': exaggeration,
                        'random_state': seed,
                        'method': 'barnes_hut',
                        'angle': theta,
                        'verbose': 0
                    }
                    tsne = SKTSNE(**params)
                    emb = tsne.fit_transform(data_proc)
                    current_iter += step
                    iterations.append(current_iter)
                    errors.append(getattr(tsne, 'kl_divergence_', float('nan')))
                    embeddings.append(emb)
                    current_init = emb
                    exaggeration = 1.0
                transformed = embeddings[-1]
                final_cost = errors[-1]
                frames = []
                frame_iters = []
                for i in range(len(embeddings) - 1):
                    start = embeddings[i]
                    end = embeddings[i + 1]
                    it_start = iterations[i]
                    it_end = iterations[i + 1]
                    for t in np.linspace(0, 1, 5, endpoint=False):
                        frames.append(start * (1 - t) + end * t)
                        frame_iters.append(int(it_start + (it_end - it_start) * t))
            else:
                params = {
                    'n_components': 2,
                    'perplexity': perplexity,
                    'learning_rate': learning_rate,
                    'metric': metric,
                    iter_kw: max_iter,
                    'init': 'random',
                    'random_state': seed,
                    'method': 'barnes_hut',
                    'angle': theta,
                    'verbose': 0
                }
                tsne = SKTSNE(**params)
                transformed = tsne.fit_transform(data_proc)
                final_cost = getattr(tsne, 'kl_divergence_', float('nan'))
                iterations = errors = frames = frame_iters = None

        gif_path = None
        feedback.pushInfo('t-SNE:')
        feedback.pushInfo('final cost:{:.6f}'.format(final_cost))

        if show_evolution and iterations:
            for it, err in zip(reversed(iterations), reversed(errors)):
                feedback.pushInfo('Iteration {}: error is {:.6f}'.format(it, err))
            # build animation from collected embeddings excluding final result
            try:
                if frames and len(frames) > 1:
                    xs = np.concatenate([e[:, 0] for e in frames])
                    ys = np.concatenate([e[:, 1] for e in frames])
                    x_min, x_max = xs.min(), xs.max()
                    y_min, y_max = ys.min(), ys.max()
                    fig, ax = plt.subplots(figsize=(6, 4))
                    scat = ax.scatter(frames[0][:, 0], frames[0][:, 1], s=10)
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)
                    ax.set_title('Iteration {}'.format(frame_iters[0]))

                    def update(frame):
                        scat.set_offsets(frames[frame])
                        idx = min(frame, len(frame_iters) - 1)
                        ax.set_title(f'Iteration {frame_iters[idx]}')
                        return scat,

                    ani = FuncAnimation(fig, update, frames=len(frames), interval=200, blit=True)
                    gif_path = os.path.join(tempfile.gettempdir(), 'tsne_animation.gif')
                    ani.save(gif_path, writer=PillowWriter(fps=10))
                    plt.close(fig)
            except Exception:
                gif_path = None

        feedback.pushInfo('Using no_dims = 2, perplexity = {}, and theta = {}'.format(2, perplexity, theta))
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

        plot_url = 'file://' + plot_path
        html = ['<html><head><meta charset="utf-8"/></head><body>',
                f'<img src="{plot_url}" alt="t-SNE Scatter Plot" style="width:100%;height:auto;"/>']
        if gif_path:
            gif_url = 'file://' + gif_path
            html.append(f'<p>Iteration Animation:</p><img src="{gif_url}" '
                       'alt="t-SNE Animation" style="width:100%;height:auto;"/>')
        html.append('</body></html>')
        report_path = os.path.join(tempfile.gettempdir(), 'tsne_report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html))

        return {self.OUTPUT: dest_id, self.REPORT: report_path}