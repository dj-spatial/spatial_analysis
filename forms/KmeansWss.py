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
from scipy.cluster.vq import kmeans, whiten, vq
import plotly
import plotly as plt
import plotly.graph_objs as go
from qgis.PyQt import uic
from qgis.core import Qgis, QgsMessageLog, QgsNetworkAccessManager, QgsProcessingUtils 
from qgis.gui import QgsMessageBar
from PyQt5.QtCore import QDate
from qgis.PyQt.QtWidgets import QVBoxLayout
from qgis.PyQt.QtWebKit import QWebSettings
from qgis.PyQt.QtWebKitWidgets import QWebView
from qgis.PyQt.QtCore import QUrl, Qt
from PyQt5.QtWidgets import QMessageBox

pluginPath = os.path.dirname(__file__)
WIDGET, BASE = uic.loadUiType(
    os.path.join(pluginPath, 'KmeansWss.ui'))


class WssWidget(BASE, WIDGET):

    def __init__(self):
        super(WssWidget, self).__init__(None)
        self.setupUi(self)

        # load the webview of the plot
        self.wss_webview_layout = QVBoxLayout()
        self.wss_webview_layout.setContentsMargins(0,0,0,0)
        self.wss_panel.setLayout(self.wss_webview_layout)
        self.wss_webview = QWebView()
        self.wss_webview.page().setNetworkAccessManager(QgsNetworkAccessManager.instance())
        wss_webview_settings = self.wss_webview.settings()
        wss_webview_settings.setAttribute(QWebSettings.WebGLEnabled, True)
        wss_webview_settings.setAttribute(QWebSettings.DeveloperExtrasEnabled, True)
        wss_webview_settings.setAttribute(QWebSettings.Accelerated2dCanvasEnabled, True)
        self.wss_webview_layout.addWidget(self.wss_webview)

        # Connect signals
        self.wssBtn.clicked.connect(self.plotView)
        self.browserBtn.clicked.connect(self.browserVeiw)

    def setSource(self, source):
        if not source:
            return
        self.source = source

    def setOptions(self, options):
        self.options = options

    def getWss(self):
        cLayer =  self.source
        to_cluster, variable_fields, normalized = self.options
        maxK = self.maxK.value()
        # input --> numpy array
        if to_cluster == 'geom':
            features = [[f.geometry().centroid().asPoint().x(), f.geometry().centroid().asPoint().y()] for f in cLayer.getFeatures()]
            features = np.stack(features, axis = 0)
        else:
            features = [[f[vf] for f in cLayer.getFeatures()] for vf in variable_fields]
            features = np.stack(features, axis = 1)

        if normalized:
            features = whiten(features)
        if maxK > features.shape[0]:
            msg = u'Clusters should be less than feature count.'
            return msg
            
        wss = []
        for i in range(0, maxK):
            codebook = kmeans(features, i+1)[0]
            distortion = vq(features, codebook)[1]
            wss.append(np.sum(distortion**2))
        diff = np.diff(wss)
        diff = np.append(0, -diff)
        diff_ratio = diff / wss[0] * 100
        diff_ratio = [("%0.1f"%i+"%") for i in diff_ratio]

        #그래프
        x_range = [i for i in range(1, maxK+1)]
        x_axis = {
            'title': 'Clusters(K)',
            'dtick': 1,
            'side': 'bottom',
            'color': 'gray',
            'zeroline': True,
            'showline': True,
            'showgrid': False,
            'autorange': True,
            'rangemode': 'nonnegative',
            'visible': True
        }
        y_axis = {
            'type': '-',
            'ticks': '',
            'zeroline': False,
            'showline': False,
            'showgrid': False,
            'autorange': True,
            'rangemode': 'nonnegative',
            'visible': False,
            'mirror':False
        }
        wssHover = ['K = ' + str(k) + '<br>' + 'WSS = ' + '{0:,.0f}'.format(w) for k, w in zip(x_range, wss)]
        
        trace0 = go.Scatter(
            x=x_range[1:],
            y=wss[1:],
            text=wssHover[1:],
            mode="lines+markers",
            line=dict(color='orange'),
            marker=dict(color='white', size=10, line=dict(color='orange', width=2)),
            textposition='top center',
            hoverinfo='text',
            name='WSS'
        )
        trace1 = go.Bar(
            x=x_range[1:],
            y=diff[1:],
            text=diff_ratio[1:],
            hoverinfo='none',
            marker=dict(color='white', line=dict(width=1, color='gray')),
            textposition='outside',
            name='ΔWSS÷TSS'
        )

        data = [trace0, trace1]
        layout = {
            'plot_bgcolor': 'white',
            'hovermode': 'closest',
            'hoverlabel': dict(font=dict(color='black')),
            'legend': dict(x=0.8, y=1, borderwidth = 0, orientation='v', traceorder='normal', tracegroupgap = 5, font={'size':12}),
            'showlegend': True,
            'xaxis1': dict(x_axis, **dict(domain=[0.0, 1.0], anchor='y1')),
            'yaxis1': dict(y_axis, **dict(domain=[0.0, 1.0], anchor='x1')),
            'margin': dict(l=0, r=0, t=15, b=45)
        }
        self.fig = {'data' : data, 'layout' : layout}
        msg = 'Success'
        return msg
        
    def plotView(self):
        msg = self.getWss()
        if  msg != 'Success':
            QMessageBox.information(self, u"Input Error", msg)
        else:
            config = {'scrollZoom': True, 'editable': False, 'displayModeBar': False}
            raw_plot = plotly.offline.plot(self.fig, output_type='div', config=config, show_link = False)
            plot_path = os.path.join(tempfile.gettempdir(), 'wss'+'.html')
            with open(plot_path, "w") as f:
                f.write(raw_plot)
            widget_layout = self.wss_webview_layout
            webview =  self.wss_webview
            plot_url = QUrl.fromLocalFile(plot_path)
            webview.load(plot_url)
            widget_layout.addWidget(webview)

    def browserVeiw(self):
        msg = self.getWss()
        if  msg != 'Success':
            QMessageBox.information(self, u"Input Error", msg)
        else:
            plt.offline.plot(self.fig, filename = os.path.join(tempfile.gettempdir(), 'wss'+'.html') , auto_open=True)

    def setValue(self, value):
        return True

    def value(self):
        return 1

class WssWidgetWrapper(WidgetWrapper):

    def __init__(self, param, dialog, row=3, col=0, **kwargs):
        super().__init__(param, dialog, row, col, **kwargs)
        self.context = dataobjects.createContext()

    def _panel(self):
        return WssWidget()

    def createWidget(self):
        if self.dialogType == DIALOG_STANDARD:
            return self._panel()

    def postInitialize(self, wrappers):
        if self.dialogType != DIALOG_STANDARD:
            return
        for wrapper in wrappers:
            if wrapper.parameterDefinition().name() == self.param.layer_param:
                self.setSource(wrapper.parameterValue())
                wrapper.widgetValueHasChanged.connect(self.layerChanged)
            elif wrapper.parameterDefinition().name() == self.param.variable_options:
                self.setOptions(wrapper.parameterValue())
                wrapper.widgetValueHasChanged.connect(self.optionsChanged)

    def layerChanged(self, wrapper):
        self.setSource(wrapper.parameterValue())

    def setSource(self, source):
        source = QgsProcessingUtils.variantToSource(source, self.context)
        self.widget.setSource(source)

    def optionsChanged(self, wrapper):
        self.setOptions(wrapper.parameterValue())

    def setOptions(self, options):
        self.widget.setOptions(options)

    def setValue(self, value):
        return self.widget.setValue(value)

    def value(self):
        return self.widget.value()

