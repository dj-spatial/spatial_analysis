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
from scipy.cluster.vq import kmeans,vq
import plotly
import plotly as plt
import plotly.graph_objs as go
from qgis.PyQt import uic
from qgis.core import Qgis, QgsMessageLog, QgsNetworkAccessManager, QgsProject 
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

    def getWss(self):
        cLayer =  QgsProject.instance().mapLayer(self.vid)
        if cLayer is None:
            msg = u'No Layer Selected'
            return msg

        # get coordinates of point features
        pts=[f.geometry().asPoint() for f in cLayer.getFeatures()]            
        x=[pt[0] for pt in pts]
        y=[pt[1] for pt in pts]
        coords = np.stack([x, y], axis = -1)
        if self.maxK > len(coords):
            msg = u'Clusters should be less than feature count.'
            return msg

        # get wss for each cluster size
        wss = []
        for i in range(1, self.maxK+1):
            centroids,_ = kmeans(coords, i)  
            cluster, distance = vq(coords, centroids)
            wss.append(sum(distance**2))

        diff = np.diff(wss)
        diff = np.append(0, -diff)
        diff_ratio = diff / wss[0] * 100
        diff_ratio = [("%0.1f"%i+"%") for i in diff_ratio]

        #그래프
        x_range = [i for i in range(1, self.maxK+1)]
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
        trace0 = go.Bar(
            x=x_range[1:],
            y=wss[1:],
            marker=dict(color='white', line=dict(width=1, color='gray')),
            text=wssHover[1:],
            hoverinfo='text',
            name='WSS'
        )
        trace1 = go.Scatter(
            x=x_range[1:],
            y=diff[1:],
            text=diff_ratio[1:],
            hoverinfo='none',
            mode="lines+markers+text",
            textposition='top center',
            name='Decline Rate'
        )
        data = [trace0, trace1]
        layout = {
            'plot_bgcolor': 'gray',
            'hovermode': 'closest',
            'hoverlabel': dict(font=dict(color='black')),
            'legend': dict(x=0.90, y=1, borderwidth = 0, orientation='v', traceorder='normal', tracegroupgap = 5, font={'size':12}),
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
        
    def setLayer(self, layer):
        self.vid = layer
        
    def setMax(self, k):
        self.maxK = k

    def value(self):
        return [1]

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
                self.setLayer(wrapper.parameterValue())
                wrapper.widgetValueHasChanged.connect(self.layerChanged)
            elif wrapper.parameterDefinition().name() == self.param.max_param:
                self.setMax(wrapper.parameterValue())
                wrapper.widgetValueHasChanged.connect(self.maxChanged)

    def layerChanged(self, wrapper):
        self.setLayer(wrapper.parameterValue())

    def setLayer(self, layer):
        self.widget.setLayer(layer)

    def maxChanged(self, wrapper):
        self.setMax(wrapper.parameterValue())

    def setMax(self, k):
        self.widget.setMax(k)

    def value(self):
        return self.widget.value()

