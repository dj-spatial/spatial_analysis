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
__date__ = 'March 2019'
__copyright__ = '(C) 2019, D.J Paek'

# This will get replaced with a git SHA1 when you do a git archive

__revision__ = '$Format:%H$'

from processing.gui.wrappers import WidgetWrapper, DIALOG_STANDARD
from processing.tools import dataobjects
import os
import tempfile
import numpy as np
from scipy.spatial import KDTree
import plotly
import plotly as plt
import plotly.graph_objs as go
from qgis.PyQt import uic
from qgis.core import QgsNetworkAccessManager, QgsProject 
from qgis.PyQt.QtWidgets import QVBoxLayout
from qgis.PyQt.QtWebKit import QWebSettings
from qgis.PyQt.QtWebKitWidgets import QWebView
from qgis.PyQt.QtCore import QUrl
from PyQt5.QtWidgets import QMessageBox

pluginPath = os.path.dirname(__file__)
WIDGET, BASE = uic.loadUiType(
    os.path.join(pluginPath, 'DbscanKnn.ui'))

class KnnWidget(BASE, WIDGET):

    def __init__(self):
        super(KnnWidget, self).__init__(None)
        self.setupUi(self)

        # load the webview of the plot
        self.knn_webview_layout = QVBoxLayout()
        self.knn_webview_layout.setContentsMargins(0,0,0,0)
        self.knn_panel.setLayout(self.knn_webview_layout)
        self.knn_webview = QWebView()
        self.knn_webview.page().setNetworkAccessManager(QgsNetworkAccessManager.instance())
        knn_webview_settings = self.knn_webview.settings()
        knn_webview_settings.setAttribute(QWebSettings.WebGLEnabled, True)
        knn_webview_settings.setAttribute(QWebSettings.DeveloperExtrasEnabled, True)
        knn_webview_settings.setAttribute(QWebSettings.Accelerated2dCanvasEnabled, True)
        self.knn_webview_layout.addWidget(self.knn_webview)

        # Connect signals
        self.knnBtn.clicked.connect(self.plotView)
        self.browserBtn.clicked.connect(self.browserVeiw)

    def getKnn(self):
        cLayer =  QgsProject.instance().mapLayer(self.vid)
        if cLayer is None:
            msg = u'Layer failed to load!'
            return msg

        # get coordinates of point features
        pts=[f.geometry().asPoint() for f in cLayer.getFeatures()]            
        x=[pt[0] for pt in pts]
        y=[pt[1] for pt in pts]
        coords = np.stack([x, y], axis = -1)
        if self.K > len(coords):
            msg = u'More clusters than feature counts.'
            return msg

        ## get distance tree
        tree = KDTree(coords)

        ## distance and index of k nearest neighbors from each point
        dist, idx = tree.query(coords, self.K+1)
        knndist = dist[:, self.K]

        knndist = np.sort(knndist)
        
        #그래프
        x_range = np.arange(1, len(knndist) + 1)
        knnHover = ['{0:,.3f}'.format(k) for k in list(knndist)]
        trace0 = go.Scatter(
            x=x_range,
            y=knndist,
            text=knnHover,
            mode="lines",
            line=dict(color='orange'),
            hoverinfo='text',
            hoverlabel=dict(bgcolor='white'),
            textposition='top center',
            name=u'Epsilon Distance to ' + str(self.K) + u' nearest neighbors'
        )    
        x_axis = {
            'title': u'Nth Point',
            'side': 'bottom',
            'zeroline': True,
            'showline': True,
            'showgrid': True,
            'autorange': True,
            'rangemode': 'nonnegative',
            'visible': True
        }
        y_axis = {
            'type': '-',
            'title': u'Epsilon Distance to ' + str(self.K) + u' nearest neighbors',
            'ticks': '',
            'zeroline': True,
            'showline': True,
            'showgrid': True,
            'autorange': True,
            'rangemode': 'nonnegative',
            'visible': True,
            'mirror':False
        }
        layout = {
            'plot_bgcolor': 'black',
            'hovermode': 'closest',
            'hoverlabel': dict(font=dict(color='black')),
            'legend': dict(x=0.80, y=1, borderwidth = 0, orientation='v', traceorder='normal', tracegroupgap = 5, font={'size':12}),
            'showlegend': False,
            'xaxis1': dict(x_axis, **dict(domain=[0.0, 1.0], anchor='y1')),
            'yaxis1': dict(y_axis, **dict(domain=[0.0, 1.0], anchor='x1')),
            'margin': dict(l=30, r=15, t=15, b=45)
        }
        data = [trace0]
        self.fig = {'data' : data, 'layout' : layout}
        msg = 'Success'
        return msg
        
    def plotView(self):
        msg = self.getKnn()
        if  msg != 'Success':
            QMessageBox.information(self, u"Input Error", msg)
        else:
            config = {'scrollZoom': True, 'editable': False, 'displayModeBar': False}
            raw_plot = plotly.offline.plot(self.fig, output_type='div', config=config, show_link = False)
            plot_path = os.path.join(tempfile.gettempdir(), 'knn'+'.html')
            with open(plot_path, "w") as f:
                f.write(raw_plot)
            widget_layout = self.knn_webview_layout
            webview =  self.knn_webview
            plot_url = QUrl.fromLocalFile(plot_path)
            webview.load(plot_url)
            widget_layout.addWidget(webview)

    def browserVeiw(self):
        msg = self.getKnn()
        if  msg != 'Success':
            QMessageBox.information(self, u"Input Error", msg)
        else:
            plt.offline.plot(self.fig, filename = os.path.join(tempfile.gettempdir(), 'knn'+'.html') , auto_open=True)
        
    def setLayer(self, layer):
        self.vid = layer
        
    def setK(self, k):
        self.K = k

    def value(self):
        return [1]

class KnnWidgetWrapper(WidgetWrapper):

    def __init__(self, param, dialog, row=3, col=0, **kwargs):
        super().__init__(param, dialog, row, col, **kwargs)
        self.context = dataobjects.createContext()

    def _panel(self):
        return KnnWidget()

    def createWidget(self):
        if self.dialogType == DIALOG_STANDARD:
            return self._panel()

    def postInitialize(self, wrappers):
        if self.dialogType != DIALOG_STANDARD:
            return
        for wrapper in wrappers:
            if wrapper.parameterDefinition().name() == self.param.layer_param:
                self.setLayer(wrapper.parameterValue())
                wrapper.widgetValueHasChanged.connect(self.layerChanged)
            elif wrapper.parameterDefinition().name() == self.param.k_param:
                self.setK(wrapper.parameterValue())
                wrapper.widgetValueHasChanged.connect(self.kChanged)

    def layerChanged(self, wrapper):
        self.setLayer(wrapper.parameterValue())

    def setLayer(self, layer):
        self.widget.setLayer(layer)

    def kChanged(self, wrapper):
        self.setK(wrapper.parameterValue())

    def setK(self, k):
        self.widget.setK(k)

    def value(self):
        return self.widget.value()

