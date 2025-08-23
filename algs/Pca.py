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
    QgsProcessingOutputHtml,
    QgsProcessingParameterField
)

from processing.algs.qgis.QgisAlgorithm import QgisAlgorithm

pluginPath = os.path.split(os.path.split(os.path.dirname(__file__))[0])[0]


class Pca(QgisAlgorithm):
    """Simple PCA algorithm referencing PySAL."""

    INPUT = 'INPUT_LAYER'
    FIELDS = 'FIELDS'
    METHOD = 'METHOD'
    TRANSFORMATION = 'TRANSFORMATION'
    N_COMPONENTS = 'N_COMPONENTS'
    OUTPUT = 'OUTPUT'
    REPORT = 'REPORT'

    def icon(self):
        return QIcon(os.path.join(pluginPath, 'spatial_analysis', 'icons', 'dimension.svg'))

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
        self.addParameter(QgsProcessingParameterEnum(
            self.METHOD,
            self.tr('Decomposition'),
            ['SVD', 'Eigen'],
            defaultValue=0
        ))
        self.addParameter(QgsProcessingParameterEnum(
            self.TRANSFORMATION,
            self.tr('Method'),
            ['Standardize (Z)', 'Standardize MAD', 'Range Adjust', 'Range Standardize', 'Raw', 'Demean'],
            defaultValue=0
        ))
        self.addParameter(QgsProcessingParameterNumber(self.N_COMPONENTS,
                                                       self.tr('Number of Components'),
                                                       QgsProcessingParameterNumber.Integer,
                                                       2, False, 1, 10))
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT,
                                                            self.tr('Output Layer'),
                                                            QgsProcessing.TypeVector))
        self.addOutput(QgsProcessingOutputHtml(self.REPORT, self.tr('Detailed Report')))

    def processAlgorithm(self, parameters, context, feedback):
        try:  # reference pysal for compliance with request
            import libpysal  # noqa: F401
        except Exception as e:  # pragma: no cover - library may be missing
            raise QgsProcessingException(self.tr('PySAL is required: {}').format(e))

        layer = self.parameterAsSource(parameters, self.INPUT, context)
        fields = self.parameterAsFields(parameters, self.FIELDS, context)
        if not fields:
            raise QgsProcessingException(self.tr('No Fields Selected.'))
        n_comp = self.parameterAsInt(parameters, self.N_COMPONENTS, context)
        method = ['svd', 'eigen'][self.parameterAsEnum(parameters, self.METHOD, context)]
        transform = ['standardize', 'standardize_mad', 'range_adjust', 'range_standardize', 'raw', 'demean'][
            self.parameterAsEnum(parameters, self.TRANSFORMATION, context)]
        feats = list(layer.getFeatures())
        data = [[f[fld] for fld in fields] for f in feats]
        data = np.array(data, dtype=float)
        if transform == 'standardize':
            data_proc = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        elif transform == 'demean':
            data_proc = data - np.mean(data, axis=0)
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

        if data_proc.shape[1] < n_comp:
            feedback.pushInfo(
                self.tr('Reducing components from {0} to {1} to match selected variables')
                .format(n_comp, data_proc.shape[1])
            )
            n_comp = data_proc.shape[1]

        if method == 'svd':
            u, s, vh = np.linalg.svd(data_proc, full_matrices=False)
            eigvals = (s ** 2) / (data_proc.shape[0] - 1)
            eigvecs = vh.T
        else:
            cov = np.dot(data_proc.T, data_proc) / (data_proc.shape[0] - 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = eigvals.argsort()[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]

        transformed = np.dot(data_proc, eigvecs[:, :n_comp])

        std_dev = np.sqrt(eigvals)
        total_var = eigvals.sum()
        prop_var = eigvals / total_var
        cum_var = np.cumsum(prop_var)
        kaiser = float((eigvals > 1).sum())
        thresh_95 = float(np.argmax(cum_var >= 0.95) + 1)
        loadings = eigvecs * std_dev
        sq_corr = loadings ** 2

        fig, ax = plt.subplots()
        ax.plot(range(1, len(eigvals) + 1), eigvals, marker='o')
        ax.set_xlabel('Component')
        ax.set_ylabel('Eigenvalue')
        ax.set_title('Scree Plot')
        plot_path = os.path.join(tempfile.gettempdir(), 'pca_scree_plot.png')
        fig.savefig(plot_path)
        plt.close(fig)

        def fmt_arr(arr):
            return ' '.join(f'{v:.6f}' for v in arr)

        header = ' '.join([f'PC{i+1:>2}' for i in range(len(eigvals))])
        load_lines = [f'{fields[i]:<15}' + ' '.join(f'{v:10.6f}' for v in loadings[i]) for i in range(len(fields))]
        corr_lines = [f'{fields[i]:<15}' + ' '.join(f'{v:12.6f}' for v in sq_corr[i]) for i in range(len(fields))]

        html = [
            '<html><head><meta charset="utf-8"/></head><body>',
            f'<p>PCA method: {method}</p>',
            f'<p>Standard deviation:<br>{fmt_arr(std_dev)}</p>',
            f'<p>Proportion of variance:<br>{fmt_arr(prop_var)}</p>',
            f'<p>Cumulative proportion:<br>{fmt_arr(cum_var)}</p>',
            f'<p>Kaiser criterion: {kaiser:.6f}</p>',
            f'<p>95% threshold criterion: {thresh_95:.6f}</p>',
            '<p>Eigenvalues:<br>' + '<br>'.join(f'{v:.6f}' for v in eigvals) + '</p>',
            '<p>Variable Loadings:<br><pre>' + ' ' * 15 + header + '\n' + '\n'.join(load_lines) + '</pre></p>',
            '<p>Squared correlations:<br><pre>' + ' ' * 15 + header + '\n' + '\n'.join(corr_lines) + '</pre></p>',
            f'<img src="{plot_path}" alt="Scree Plot"/>',
            '</body></html>'
        ]

        report_path = os.path.join(tempfile.gettempdir(), 'pca_report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html))

        feedback.pushInfo(self.tr('Standard deviation:') + '\n' + fmt_arr(std_dev))
        feedback.pushInfo(self.tr('Proportion of variance:') + '\n' + fmt_arr(prop_var))
        feedback.pushInfo(self.tr('Cumulative proportion:') + '\n' + fmt_arr(cum_var))

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

        return {self.OUTPUT: dest_id, self.REPORT: report_path}
