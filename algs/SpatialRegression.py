# -*- coding: utf-8 -*-
"""
/***************************************************************************
                                 A QGIS plugin
SpatialAnalyzer
                              -------------------
        git sha              : $Format:%H$
        copyright            : (C) 2024 by D.J Paek
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

from qgis.PyQt.QtCore import QVariant, QUrl
from qgis.PyQt.QtGui import QIcon

from qgis.core import (
    QgsField,
    QgsFields,
    QgsProcessing,
    QgsProcessingException,
    QgsProcessingParameterEnum,
    QgsProcessingParameterField,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterString,
    QgsProcessingParameterFileDestination,
    QgsProcessingUtils,
    QgsFeature,
    QgsFeatureSink,
)

from processing.algs.qgis.QgisAlgorithm import QgisAlgorithm

# pysal modules - may not be available
try:  # pragma: no cover - library might be missing
    from spreg import OLS, ML_Lag, ML_Error
except Exception:  # pragma: no cover - runtime import
    OLS = ML_Lag = ML_Error = None

pluginPath = os.path.split(os.path.split(os.path.dirname(__file__))[0])[0]


class SpatialRegression(QgisAlgorithm):

    INPUT = 'INPUT_LAYER'
    DEPENDENT = 'DEPENDENT'
    INDEPENDENTS = 'INDEPENDENTS'
    MODEL = 'MODEL'
    WEIGHTS_BTN = 'WEIGHTS_BTN'
    OUTPUT = 'OUTPUT'
    OUTPUT_REPORT = 'OUTPUT_REPORT'
    WEIGHT_REPORT = 'WEIGHT_REPORT'

    def icon(self):
        return QIcon(os.path.join(pluginPath, 'spatial_analysis', 'icons', 'central.svg'))

    def group(self):
        return self.tr('Spatial Regression')

    def groupId(self):
        return 'spatialregression'

    def name(self):
        return 'spatialregression'

    def displayName(self):
        return self.tr('Spatial Regression')

    def __init__(self):
        super().__init__()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.INPUT,
            self.tr('Input Layer'),
            [QgsProcessing.TypeVector]))

        self.addParameter(QgsProcessingParameterEnum(
            self.MODEL,
            self.tr('Model Type'),
            options=[self.tr('Classic'), self.tr('Spatial Lag Model'), self.tr('Spatial Error Model')],
            defaultValue=0))

        self.addParameter(QgsProcessingParameterField(
            self.DEPENDENT,
            self.tr('Dependent Variable'),
            parentLayerParameterName=self.INPUT,
            type=QgsProcessingParameterField.Numeric))

        self.addParameter(QgsProcessingParameterField(
            self.INDEPENDENTS,
            self.tr('Independent Variables'),
            parentLayerParameterName=self.INPUT,
            type=QgsProcessingParameterField.Numeric,
            allowMultiple=True))

        weights_param = QgsProcessingParameterString(
            self.WEIGHTS_BTN,
            self.tr('Weights'),
            '',
            optional=False)
        weights_param.setMetadata({'widget_wrapper': {
            'class': 'spatial_analysis.forms.WeightsWidget.WeightsWidgetWrapper',
            'layer_param': self.INPUT}})
        self.addParameter(weights_param)

        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT,
            self.tr('Output Layer'),
            QgsProcessing.TypeVector))

        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT_REPORT,
            self.tr('Output Report'),
            'HTML files (*.html)'))

        self.addParameter(QgsProcessingParameterFileDestination(
            self.WEIGHT_REPORT,
            self.tr('Weight Report'),
            'HTML files (*.html)',
            optional=True))

    def processAlgorithm(self, parameters, context, feedback):
        if OLS is None:
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
        dep_field = self.parameterAsString(parameters, self.DEPENDENT, context)
        ind_fields = self.parameterAsFields(parameters, self.INDEPENDENTS, context)
        model_idx = self.parameterAsEnum(parameters, self.MODEL, context)
        weight_info = parameters.get(self.WEIGHTS_BTN)

        import numpy as np

        if not weight_info:
            raise QgsProcessingException(self.tr('Weights must be defined.'))

        if model_idx == 0:
            feats = list(layer.getFeatures())
            y = np.array([f[dep_field] for f in feats]).reshape((-1, 1))
            X = np.array([[f[fld] for fld in ind_fields] for f in feats])
            model = OLS(y, X, name_y=dep_field, name_x=ind_fields)
        else:
            w = weight_info['weights']
            id_field = weight_info['id_field']
            id_to_feat = {}
            id_to_y = {}
            id_to_x = {}
            for f in layer.getFeatures():
                fid = f[id_field]
                id_to_feat[fid] = f
                id_to_y[fid] = f[dep_field]
                id_to_x[fid] = [f[fld] for fld in ind_fields]
            try:
                feats = [id_to_feat[i] for i in w.id_order]
                y = np.array([id_to_y[i] for i in w.id_order]).reshape((-1, 1))
                X = np.array([id_to_x[i] for i in w.id_order])
            except KeyError:
                raise QgsProcessingException(self.tr('ID field mismatch between weights and layer.'))
            w.transform = 'r'
            if model_idx == 1:
                model = ML_Lag(y, X, w, name_y=dep_field, name_x=ind_fields)
            else:
                model = ML_Error(y, X, w, name_y=dep_field, name_x=ind_fields)

        predictions = model.predy.flatten()
        actual = y.flatten()
        residuals = actual - predictions
        pred_errors = predictions - actual


        fields = layer.fields()
        new_fields = QgsFields()
        new_fields.append(QgsField('Predicted', QVariant.Double))
        new_fields.append(QgsField('Residual', QVariant.Double))
        new_fields.append(QgsField('Pred Error', QVariant.Double))
        fields = QgsProcessingUtils.combineFields(fields, new_fields)

        (sink, dest_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            fields,
            layer.wkbType(),
            layer.sourceCrs())

        for idx, f in enumerate(feats):
            out_f = QgsFeature(fields)
            out_f.setGeometry(f.geometry())
            attrs = f.attributes() + [float(predictions[idx]), float(residuals[idx]), float(pred_errors[idx])]
            out_f.setAttributes(attrs)
            sink.addFeature(out_f, QgsFeatureSink.FastInsert)

        output_report = self.parameterAsFileOutput(parameters, self.OUTPUT_REPORT, context)
        summary_text = (
            model.summary.as_text() if hasattr(model.summary, 'as_text')
            else str(model.summary)
        )

        def _section(text, start, *end_markers):
            upper = text.upper()
            start_idx = upper.find(start)
            if start_idx == -1:
                return ''
            start_idx += len(start)
            end_idx = len(text)
            for end in end_markers:
                idx = upper.find(end, start_idx)
                if idx != -1 and idx < end_idx:
                    end_idx = idx
            return text[start_idx:end_idx].strip()

        regression_txt = _section(summary_text, 'REGRESSION', 'REGRESSION DIAGNOSTICS')
        reg_diag_txt = _section(summary_text, 'REGRESSION DIAGNOSTICS', 'DIAGNOSTICS FOR SPATIAL DEPENDENCE')
        spatial_dep_txt = _section(summary_text, 'DIAGNOSTICS FOR SPATIAL DEPENDENCE', 'OBS')

        vm = getattr(model, 'vm', None)
        matrix_lines = None
        if vm is not None:
            names = list(ind_fields)
            if vm.shape[0] == len(ind_fields) + 1:
                names = ['Intercept'] + names
            matrix_lines = ['\t'.join([''] + names)]
            for name, row in zip(names, vm):
                row_str = '\t'.join('{0:.6f}'.format(val) for val in row)
                matrix_lines.append('\t'.join([name, row_str]))

        obs_lines = ['OBS\t{0}\tPREDICTED\tRESIDUAL\tPRED ERROR'.format(dep_field)]
        for idx, val in enumerate(actual, start=1):
            pred = predictions[idx - 1]
            resid = residuals[idx - 1]
            pred_err = pred_errors[idx - 1]
            obs_lines.append(
                '{0}\t{1:.5f}\t{2:.5f}\t{3:.5f}\t{4:.5f}'.format(
                    idx, val, pred, resid, pred_err
                )
            )

        with codecs.open(output_report, 'w', encoding='utf-8') as f:
            f.write('<html><head>\n')
            f.write('<meta http-equiv="Content-Type" content="text/html; charset=utf-8" /></head><body>\n')
            if regression_txt:
                f.write('<h2>{0}</h2>\n<pre>{1}</pre>\n'.format(self.tr('Regression'), regression_txt))
            if reg_diag_txt:
                f.write('<h2>{0}</h2>\n<pre>{1}</pre>\n'.format(self.tr('Regression Diagnostics'), reg_diag_txt))
            if spatial_dep_txt:
                f.write('<h2>{0}</h2>\n<pre>{1}</pre>\n'.format(self.tr('Diagnostics for Spatial Dependence'), spatial_dep_txt))
            if matrix_lines:
                f.write('<h2>{0}</h2>\n<pre>{1}</pre>\n'.format(self.tr('Coefficients Variance Matrix'), '\n'.join(matrix_lines)))
            f.write('<h2>{0}</h2>\n<pre>{1}</pre>\n'.format(self.tr('Observations'), '\n'.join(obs_lines)))
            f.write('</body></html>')

        weight_report = self.parameterAsFileOutput(parameters, self.WEIGHT_REPORT, context) if weight_info else ''
        if weight_info and weight_report:
            summary = weight_info.get('summary', '')
            with codecs.open(weight_report, 'w', encoding='utf-8') as f:
                f.write('<html><head>\n')
                f.write('<meta http-equiv="Content-Type" content="text/html; charset=utf-8" /></head><body>\n')
                f.write('<pre>{0}</pre>\n'.format(summary))
                f.write('</body></html>')

        results = {self.OUTPUT: dest_id, self.OUTPUT_REPORT: output_report}
        return results

    def checkParameterValues(self, parameters, context):
        ok, msg = super().checkParameterValues(parameters, context)
        if not ok:
            return ok, msg
        if not parameters.get(self.WEIGHTS_BTN):
            return False, self.tr('Weights must be defined.')
        return True, ''

