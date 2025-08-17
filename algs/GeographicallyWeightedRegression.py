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
import numpy as np
from math import radians, pi, sqrt

from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtGui import QIcon
from qgis.core import (
    QgsFeature,
    QgsField,
    QgsFields,
    QgsProcessing,
    QgsProcessingException,
    QgsProcessingUtils,
    QgsFeatureSink,
    QgsProcessingParameterNumber,
    QgsProcessingParameterField,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterEnum,
    QgsProcessingParameterBoolean,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsProject,
)
from processing.algs.qgis.QgisAlgorithm import QgisAlgorithm

pluginPath = os.path.split(os.path.split(os.path.dirname(__file__))[0])[0]


class GeographicallyWeightedRegression(QgisAlgorithm):
    INPUT = 'INPUT_LAYER'
    DEP_FIELD = 'DEP_FIELD'
    LOCAL_FIELDS = 'LOCAL_FIELDS'
    GLOBAL_FIELDS = 'GLOBAL_FIELDS'
    MODEL = 'MODEL'
    DISTANCE = 'DISTANCE'
    KERNEL = 'KERNEL'
    BANDWIDTH = 'BANDWIDTH'
    BANDWIDTH_SEL = 'BANDWIDTH_SEL'
    CRITERION = 'CRITERION'
    STANDARDIZE = 'STANDARDIZE'
    VAR_TEST = 'VAR_TEST'
    LTOG = 'LTOG'
    GTOL = 'GTOL'
    FORCE_GLOBAL = 'FORCE_GLOBAL'
    OUTPUT = 'OUTPUT'

    MODEL_OPTS = ['Gaussian (GWR)', 'Poisson (GWPR)', 'Logistic (GWLR)']
    DIST_OPTS = ['Projected', 'Spherical']
    KERNEL_OPTS = ['Fixed Gaussian', 'Fixed bi-square',
                   'Adaptive Gaussian', 'Adaptive bi-square']
    BW_SEL_OPTS = ['Single bandwidth', 'Golden selection search',
                   'Interval search']
    CRIT_OPTS = ['AICc', 'AIC', 'BIC/MDL', 'CV']

    def icon(self):
        return QIcon(os.path.join(pluginPath, 'spatial_analysis', 'icons', 'cluster.svg'))

    def group(self):
        return self.tr('Spatial Regression')

    def groupId(self):
        return 'spatialregression'

    def name(self):
        return 'gwr'

    def displayName(self):
        return self.tr('Geographically Weighted Regression')

    def __init__(self):
        super().__init__()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.INPUT,
            self.tr('Input Layer'),
            [QgsProcessing.TypeVectorAnyGeometry]))
        self.addParameter(QgsProcessingParameterField(
            self.DEP_FIELD,
            self.tr('Dependent Field'),
            parentLayerParameterName=self.INPUT,
            type=QgsProcessingParameterField.Numeric))
        self.addParameter(QgsProcessingParameterField(
            self.LOCAL_FIELDS,
            self.tr('Local Fields'),
            parentLayerParameterName=self.INPUT,
            type=QgsProcessingParameterField.Numeric,
            allowMultiple=True,
            optional=True))
        self.addParameter(QgsProcessingParameterField(
            self.GLOBAL_FIELDS,
            self.tr('Global Fields'),
            parentLayerParameterName=self.INPUT,
            type=QgsProcessingParameterField.Numeric,
            allowMultiple=True,
            optional=True))
        self.addParameter(QgsProcessingParameterEnum(
            self.MODEL,
            self.tr('Model Type'),
            options=self.MODEL_OPTS,
            defaultValue=0))
        self.addParameter(QgsProcessingParameterEnum(
            self.DISTANCE,
            self.tr('Distance Type'),
            options=self.DIST_OPTS,
            defaultValue=0))
        self.addParameter(QgsProcessingParameterEnum(
            self.KERNEL,
            self.tr('Kernel Function'),
            options=self.KERNEL_OPTS,
            defaultValue=0))
        self.addParameter(QgsProcessingParameterNumber(
            self.BANDWIDTH,
            self.tr('Bandwidth / Neighbors'),
            QgsProcessingParameterNumber.Double,
            1.0, False, 1e-6, 1e9))
        self.addParameter(QgsProcessingParameterEnum(
            self.BANDWIDTH_SEL,
            self.tr('Bandwidth Selection'),
            options=self.BW_SEL_OPTS,
            defaultValue=0))
        self.addParameter(QgsProcessingParameterEnum(
            self.CRITERION,
            self.tr('Selection Criterion'),
            options=self.CRIT_OPTS,
            defaultValue=0))
        self.addParameter(QgsProcessingParameterBoolean(
            self.STANDARDIZE,
            self.tr('Standardisation'),
            defaultValue=False))
        self.addParameter(QgsProcessingParameterBoolean(
            self.VAR_TEST,
            self.tr('Geographical variability test'),
            defaultValue=False))
        self.addParameter(QgsProcessingParameterBoolean(
            self.LTOG,
            self.tr('L to G variable selection'),
            defaultValue=False))
        self.addParameter(QgsProcessingParameterBoolean(
            self.GTOL,
            self.tr('G to L variable selection'),
            defaultValue=False))
        self.addParameter(QgsProcessingParameterBoolean(
            self.FORCE_GLOBAL,
            self.tr('Force global coefficients to mean'),
            defaultValue=True))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT,
            self.tr('Output Layer'),
            QgsProcessing.TypeVector))

    def processAlgorithm(self, parameters, context, feedback):
        layer = self.parameterAsSource(parameters, self.INPUT, context)
        dep_field = self.parameterAsString(parameters, self.DEP_FIELD, context)
        local_fields = self.parameterAsFields(parameters, self.LOCAL_FIELDS, context)
        global_fields = self.parameterAsFields(parameters, self.GLOBAL_FIELDS, context)
        if not local_fields and not global_fields:
            raise QgsProcessingException(self.tr('At least one explanatory field is required.'))
        model = ['gaussian', 'poisson', 'logistic'][
            self.parameterAsEnum(parameters, self.MODEL, context)]
        dist_type = ['projected', 'spherical'][
            self.parameterAsEnum(parameters, self.DISTANCE, context)]
        kernel = ['fixed_gaussian', 'fixed_bisquare',
                  'adaptive_gaussian', 'adaptive_bisquare'][
            self.parameterAsEnum(parameters, self.KERNEL, context)]
        bw_sel = ['single', 'golden', 'interval'][
            self.parameterAsEnum(parameters, self.BANDWIDTH_SEL, context)]
        criterion = ['AICc', 'AIC', 'BIC', 'CV'][
            self.parameterAsEnum(parameters, self.CRITERION, context)]
        standardize = self.parameterAsBool(parameters, self.STANDARDIZE, context)
        var_test = self.parameterAsBool(parameters, self.VAR_TEST, context)
        ltog = self.parameterAsBool(parameters, self.LTOG, context)
        gtol = self.parameterAsBool(parameters, self.GTOL, context)
        force_global = self.parameterAsBool(parameters, self.FORCE_GLOBAL, context)
        bandwidth = self.parameterAsDouble(parameters, self.BANDWIDTH, context)

        if criterion == 'CV' and model != 'gaussian':
            raise QgsProcessingException(self.tr('CV criterion only supported for Gaussian model'))

        if 'adaptive' in kernel:
            if abs(bandwidth - int(bandwidth)) > 1e-9 or bandwidth < 1:
                raise QgsProcessingException(self.tr('Adaptive kernels require a positive integer neighbor count'))
            bandwidth = int(bandwidth)

        feats = list(layer.getFeatures())
        coords = []
        y = []
        X_local = []
        X_global = []

        if dist_type == 'spherical':
            crs_src = layer.sourceCrs()
            if crs_src != QgsCoordinateReferenceSystem('EPSG:4326'):
                transform = QgsCoordinateTransform(crs_src, QgsCoordinateReferenceSystem('EPSG:4326'), QgsProject.instance())
            else:
                transform = None
        else:
            transform = None

        for feat in feats:
            geom = feat.geometry()
            pt = geom.asPoint() if geom.type() == 0 else geom.centroid().asPoint()
            if transform:
                pt = transform.transform(pt)
            coords.append((pt.x(), pt.y()))
            y.append(feat[dep_field])
            if local_fields:
                X_local.append([feat[f] for f in local_fields])
            if global_fields:
                X_global.append([feat[f] for f in global_fields])

        coords = np.array(coords, dtype=float)
        y = np.array(y, dtype=float)
        if local_fields:
            X_local = np.array(X_local, dtype=float)
        else:
            X_local = np.zeros((len(feats), 0))
        if global_fields:
            X_global = np.array(X_global, dtype=float)
        else:
            X_global = np.zeros((len(feats), 0))

        if standardize:
            if X_local.size:
                means = X_local.mean(axis=0)
                stds = X_local.std(axis=0)
                stds[stds == 0] = 1
                X_local = (X_local - means) / stds
            if X_global.size:
                means = X_global.mean(axis=0)
                stds = X_global.std(axis=0)
                stds[stds == 0] = 1
                X_global = (X_global - means) / stds

        X = np.hstack([np.ones((len(feats), 1)), X_global, X_local])
        k_global = X_global.shape[1]
        k_local = X_local.shape[1]

        if bw_sel != 'single':
            bandwidth = self._select_bandwidth(coords, y, X, model, kernel,
                                               dist_type, bw_sel, criterion,
                                               bandwidth, feedback)
        betas, ses, hat, mu = self._gwr(coords, y, X, bandwidth, kernel,
                                        model, dist_type, feedback)

        # variable selection
        if k_local and ltog:
            for j in range(k_local):
                idx = 1 + k_global + j
                if np.std(betas[:, idx]) < 1e-6:
                    betas[:, idx] = betas[:, idx].mean()
        if k_global and gtol:
            for j in range(k_global):
                idx = 1 + j
                if np.std(betas[:, idx]) > 1e-6:
                    continue
                betas[:, idx] = betas[:, idx].mean()

        if k_global and force_global:
            for j in range(k_global):
                idx = 1 + j
                betas[:, idx] = betas[:, idx].mean()

        t_vals = None
        if var_test and k_local:
            t_vals = np.zeros((len(feats), k_local))
            for j in range(k_local):
                idx = 1 + k_global + j
                mean_beta = betas[:, idx].mean()
                t_vals[:, j] = (betas[:, idx] - mean_beta) / ses[:, idx]

        fields = layer.fields()
        new_fields = QgsFields()
        new_fields.append(QgsField('GWR_Int', QVariant.Double))
        for f in global_fields:
            new_fields.append(QgsField('GWR_G_{}'.format(f), QVariant.Double))
        for f in local_fields:
            new_fields.append(QgsField('GWR_L_{}'.format(f), QVariant.Double))
            if var_test:
                new_fields.append(QgsField('t_{}'.format(f), QVariant.Double))
        fields = QgsProcessingUtils.combineFields(fields, new_fields)
        (sink, dest_id) = self.parameterAsSink(parameters, self.OUTPUT, context,
                                              fields, layer.wkbType(), layer.sourceCrs())

        for idx, feat in enumerate(feats):
            attrs = feat.attributes()
            row = [float(betas[idx, 0])]
            if k_global:
                row.extend([float(b) for b in betas[idx, 1:1 + k_global]])
            if k_local:
                row.extend([float(b) for b in betas[idx, 1 + k_global:1 + k_global + k_local]])
                if var_test:
                    row.extend([float(t) for t in t_vals[idx]])
            attrs.extend(row)
            out_feat = QgsFeature()
            out_feat.setGeometry(feat.geometry())
            out_feat.setAttributes(attrs)
            sink.addFeature(out_feat, QgsFeatureSink.FastInsert)

        feedback.pushInfo(self.tr('Done with GWR'))
        return {self.OUTPUT: dest_id}

    # distance
    def _distance(self, coord, coords, dist_type):
        if dist_type == 'projected':
            diff = coords - coord
            return np.sqrt((diff ** 2).sum(axis=1))
        lon1, lat1 = radians(coord[0]), radians(coord[1])
        lon2 = np.radians(coords[:, 0])
        lat2 = np.radians(coords[:, 1])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return 6371000 * c

    def _kernel(self, d, bw, kernel):
        if 'adaptive' in kernel:
            idx = int(min(len(d) - 1, bw))
            bw = np.sort(d)[idx]
            if bw == 0:
                bw = 1e-9
        if 'gaussian' in kernel:
            w = np.exp(-0.5 * (d / bw) ** 2)
        else:
            w = (1 - (d / bw) ** 2)
            w[w < 0] = 0
            w = w ** 2
        return w

    def _gwr(self, coords, y, X, bw, kernel, model, dist_type, feedback):
        n, k = X.shape
        betas = np.zeros((n, k))
        ses = np.zeros((n, k))
        hat = np.zeros(n)
        mu = np.zeros(n)
        for i in range(n):
            d = self._distance(coords[i], coords, dist_type)
            w = self._kernel(d, bw, kernel)
            if model == 'gaussian':
                XtW = X.T * w
                XtWX = XtW @ X
                XtWX_inv = np.linalg.pinv(XtWX)
                XtWy = XtW @ y
                beta = XtWX_inv @ XtWy
                resid = y - X @ beta
                s2 = (w * resid ** 2).sum() / (w.sum() - k)
                ses[i] = np.sqrt(np.diag(XtWX_inv) * s2)
                hat[i] = X[i] @ XtWX_inv @ X[i]
                mu[i] = X[i] @ beta
            else:
                beta = np.zeros(k)
                eps = 1e-9
                for _ in range(100):
                    eta = X @ beta
                    if model == 'poisson':
                        mu_i = np.exp(eta)
                        mu_i = np.clip(mu_i, eps, None)
                        z = eta + (y - mu_i) / mu_i
                        W_irls = mu_i
                    else:
                        mu_i = 1 / (1 + np.exp(-eta))
                        mu_i = np.clip(mu_i, eps, 1 - eps)
                        z = eta + (y - mu_i) / (mu_i * (1 - mu_i))
                        W_irls = mu_i * (1 - mu_i)
                    XtW = X.T * (w * W_irls)
                    XtWX = XtW @ X
                    XtWX_inv = np.linalg.pinv(XtWX)
                    XtWy = XtW @ z
                    beta_new = XtWX_inv @ XtWy
                    if np.max(np.abs(beta_new - beta)) < 1e-5:
                        beta = beta_new
                        break
                    beta = beta_new
                if model == 'poisson':
                    mu_i = np.exp(X @ beta)
                    mu_i = np.clip(mu_i, eps, None)
                else:
                    mu_i = 1 / (1 + np.exp(-(X @ beta)))
                    mu_i = np.clip(mu_i, eps, 1 - eps)
                resid = y - mu_i
                ses[i] = np.sqrt(np.diag(XtWX_inv))
                hat[i] = np.nan
                mu[i] = mu_i[i]
            betas[i] = beta
            if feedback is not None:
                feedback.setProgress(int((i + 1) / n * 100))
        return betas, ses, hat, mu

    def _select_bandwidth(self, coords, y, X, model, kernel, dist_type,
                           method, criterion, bw, feedback):
        n = len(coords)
        if 'adaptive' in kernel:
            min_bw, max_bw = 2, n - 1
        else:
            max_dist = 0
            for i in range(n):
                d = self._distance(coords[i], coords, dist_type)
                max_dist = max(max_dist, d.max())
            min_bw, max_bw = 0.01 * max_dist, max_dist
        if method == 'single':
            return bw

        def score(bw_val):
            if 'adaptive' in kernel:
                bw_val = int(round(bw_val))
            betas, ses, hat, mu = self._gwr(coords, y, X, bw_val, kernel, model, dist_type, None)
            return self._criterion(y, mu, betas.shape[1], hat, criterion, model)

        if method == 'golden':
            phi = (1 + sqrt(5)) / 2
            a, b = min_bw, max_bw
            c = b - (b - a) / phi
            d = a + (b - a) / phi
            fc = score(c)
            fd = score(d)
            while abs(b - a) > 1e-3:
                if fc < fd:
                    b, d, fd = d, c, fc
                    c = b - (b - a) / phi
                    fc = score(c)
                else:
                    a, c, fc = c, d, fd
                    d = a + (b - a) / phi
                    fd = score(d)
            bw_opt = (b + a) / 2
        else:  # interval
            bw_vals = np.linspace(min_bw, max_bw, 20)
            scores = [score(bv) for bv in bw_vals]
            bw_opt = bw_vals[int(np.argmin(scores))]
        if 'adaptive' in kernel:
            return int(round(bw_opt))
        return bw_opt

    def _criterion(self, y, mu, k, hat, criterion, model):
        n = len(y)
        if criterion == 'CV':
            if model != 'gaussian':
                raise QgsProcessingException(self.tr('CV only supported for Gaussian model'))
            return np.sum(((y - mu) / (1 - hat)) ** 2)
        if model == 'gaussian':
            rss = np.sum((y - mu) ** 2)
            sigma2 = rss / n
            ll = -0.5 * n * (np.log(2 * pi * sigma2) + 1)
        elif model == 'poisson':
            ll = np.sum(y * np.log(mu) - mu)
        else:
            ll = np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu))
        aic = -2 * ll + 2 * k
        if criterion == 'AIC':
            return aic
        if criterion == 'AICc':
            return aic + (2 * k * (k + 1)) / (n - k - 1)
        if criterion == 'BIC':
            return -2 * ll + k * np.log(n)
        return aic
