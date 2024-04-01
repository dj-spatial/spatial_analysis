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
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram
import plotly
import plotly as plt
import plotly.graph_objs as go
import plotly.figure_factory as ff
from qgis.PyQt import uic
from qgis.core import QgsNetworkAccessManager, QgsProject 
from qgis.PyQt.QtWidgets import QVBoxLayout
from qgis.PyQt.QtWebKit import QWebSettings
from qgis.PyQt.QtWebKitWidgets import QWebView
from qgis.PyQt.QtCore import QUrl
from PyQt5.QtWidgets import QMessageBox

pluginPath = os.path.dirname(__file__)
WIDGET, BASE = uic.loadUiType(
    os.path.join(pluginPath, 'Harchi.ui'))


class HarchiWidget(BASE, WIDGET):

    def __init__(self):
        super(HarchiWidget, self).__init__(None)
        self.setupUi(self)

        # load the webview of the plot
        self.harchi_webview_layout = QVBoxLayout()
        self.harchi_webview_layout.setContentsMargins(0,0,0,0)
        self.harchi_panel.setLayout(self.harchi_webview_layout)
        self.harchi_webview = QWebView()
        self.harchi_webview.page().setNetworkAccessManager(QgsNetworkAccessManager.instance())
        harchi_webview_settings = self.harchi_webview.settings()
        harchi_webview_settings.setAttribute(QWebSettings.WebGLEnabled, True)
        harchi_webview_settings.setAttribute(QWebSettings.DeveloperExtrasEnabled, True)
        harchi_webview_settings.setAttribute(QWebSettings.Accelerated2dCanvasEnabled, True)
        self.harchi_webview_layout.addWidget(self.harchi_webview)
        self.graphType = self.comboBox.currentIndex()

        # Connect signals
        self.plotBtn.clicked.connect(self.plotView)
        self.browserBtn.clicked.connect(self.browserView)
        self.comboBox.currentIndexChanged.connect(self.showLabel)

    def showLabel(self):
        if self.comboBox.currentIndex() == 0:
            self.label.show()
            self.labelCB.show()
        else:
            self.label.hide()
            self.labelCB.hide()

    def fillLabel(self):
        cLayer =  QgsProject.instance().mapLayer(self.vid)
        if cLayer is None:
            msg = u'No Layer Selected.'
            return msg
        fields = cLayer.fields()
        self.fNames = fields.names()
        self.labelCB.addItems(self.fNames)

    def getHarchi(self):
        cLayer =  QgsProject.instance().mapLayer(self.vid)
        if cLayer is None:
            msg = u'No Layer Selected.'
            return msg

        # get coordinates of point features
        pts=[f.geometry().asPoint() for f in cLayer.getFeatures()]            
        x=[pt[0] for pt in pts]
        y=[pt[1] for pt in pts]
        coords = np.stack([x, y], axis = -1)
        dMethod = ['centroid', 'ward', 'single', 'complete', 'average']
        if self.comboBox.currentIndex() == 0:
            Z = linkage(coords, dMethod[self.linkageIdx], metric = 'euclidean')
            ddata = dendrogram(Z)
            
            labels=[f.attributes()[self.labelCB.currentIndex()] for f in cLayer.getFeatures()]
            hText=list(Z[:,2])
            dendro = ff.create_dendrogram(coords, linkagefun = lambda x : Z, labels = labels, hovertext = ddata['dcoord'])

            #그래프
            dendro['layout'].update({'autosize': True, 'title' : 'Dendrogram by ' + dMethod[self.linkageIdx] + ' method' })
            dendro['layout']['xaxis'].update({'mirror': False,
                                               'showgrid': False,
                                               'showline': False,
                                               'zeroline': False,
                                               'ticks':""})
            dendro['layout']['yaxis'].update({'mirror': False,
                                              'showgrid': False,
                                              'showline': False,
                                              'zeroline': False,
                                              'showticklabels': True,
                                              'ticks': ""})
            self.fig = {'data' : dendro}
        elif self.comboBox.currentIndex() == 1:
            #Cophenet Correlation
            cophenet_corr = []
            for m in dMethod:
                Z = linkage(coords, method = m, metric = 'euclidean')
                c,_ = cophenet(Z, pdist(coords))
                cophenet_corr.append(c)
            cophenet_corr = [("%0.3f"%i) for i in cophenet_corr]
            trace0 = go.Bar(
                x=dMethod,
                y=cophenet_corr,
                marker=dict(color='white', line=dict(width=1, color='navy')),
                text=cophenet_corr,
                textposition='outside',
                hoverinfo='text',
                name=u'Cophenet Correlation'
            )
            data = [trace0]
            layout = go.Layout(
                title = 'Cophenet Correlation',
                xaxis = dict(title='Linkage Method')
            )
            self.fig = go.Figure(data = data, layout = layout)
        else:
            #Cophenet Distance
            revDiff = [0]
            Z = linkage(coords, method = dMethod[self.linkageIdx], metric = 'euclidean')
            cDistance = Z[:,2]
            diff = np.diff(cDistance, 1)
            revDistance = cDistance[::-1]
            revDiff = np.insert(diff[::-1], 0, 0, axis=0)
            x_range = np.arange(1, len(cDistance) + 1)
            dHover = [u'Last ' + str(n) + u'th Cluster' + '<br>' + 'Cophenet Distance = ' + '{0:,.1f}'.format(d) for n, d in zip(x_range, revDistance)]
            diffHover = [u'Last ' + str(n) + u'th Cluster' + '<br>' + u'Distance Decline = ' + '{0:,.1f}'.format(d) for n, d in zip(x_range, revDiff)]

            trace0 = go.Scatter(
                x=x_range,
                y=revDistance,
                text=dHover,
                mode="lines",
                line=dict(color='orange'),
                hoverinfo='text',
                hoverlabel=dict(bgcolor='white'),
                textposition='top center',
                name=u'Cophenet Distance'
            )
            trace1 = go.Bar(
                x=x_range,
                y=revDiff,
                marker=dict(color='white', line=dict(width=1, color='white')),
                text=diffHover,
                hoverinfo='text',
                name='Distance Decline'
            )
            x_axis = {
                'title': u'Last Nth Cluster Merge',
                'dtick': 5,
                'side': 'bottom',
                'zeroline': False,
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
                'rangemode': 'normal',
                'visible': False,
                'mirror':False
            }
            y_axis2 = {
                'type': '-',
                'ticks': '',
                'zeroline': False,
                'showline': False,
                'showgrid': False,
                'autorange': True,
                'rangemode': 'normal',
                'visible': False,
                'mirror':False
            }
            layout = {
                'plot_bgcolor': 'gray',
                'hovermode': 'closest',
                'hoverlabel': dict(font=dict(color='black')),
                'legend': dict(x=0.80, y=1, borderwidth = 0, orientation='v', traceorder='normal', tracegroupgap = 5, font={'size':12}),
                'showlegend': True,
                'margin': dict(l=15, r=0, t=15, b=45)
            }
            data = [trace0, trace1]
            self.fig = go.Figure(data = data, layout = layout)
        msg = 'Success'
        return msg
        
    def plotView(self):
        msg = self.getHarchi()
        if  msg != 'Success':
            QMessageBox.information(self, u"Input Error", msg)
        else:
            config = {'scrollZoom': True, 'editable': False, 'displayModeBar': False}
            raw_plot = plotly.offline.plot(self.fig, output_type='div', config=config, show_link = False)
            plot_path = os.path.join(tempfile.gettempdir(), 'dendro'+'.html')
            with open(plot_path, "w") as f:
                f.write(raw_plot)
            widget_layout = self.harchi_webview_layout
            webview =  self.harchi_webview
            plot_url = QUrl.fromLocalFile(plot_path)
            webview.load(plot_url)
            widget_layout.addWidget(webview)

    def browserView(self):
        msg = self.getHarchi()
        if  msg != 'Success':
            QMessageBox.information(self, u"Input Error", msg)
        else:
            plt.offline.plot(self.fig, filename = os.path.join(tempfile.gettempdir(), 'harchi'+'.html') , auto_open=True)
        
    def setLayer(self, layer):
        self.vid = layer
        
    def setLinkage(self, linkage):
        self.linkageIdx = linkage

    def value(self):
        return ['Graph Output']

class HarchiWidgetWrapper(WidgetWrapper):

    def __init__(self, param, dialog, row=3, col=0, **kwargs):
        super().__init__(param, dialog, row, col, **kwargs)
        self.context = dataobjects.createContext()

    def _panel(self):
        return HarchiWidget()

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
            elif wrapper.parameterDefinition().name() == self.param.linkage_param:
                self.setLinkage(wrapper.parameterValue())
                wrapper.widgetValueHasChanged.connect(self.linkageChanged)

    def layerChanged(self, wrapper):
        self.setLayer(wrapper.parameterValue())
        self.widget.labelCB.clear()
        self.widget.fillLabel()

    def setLayer(self, layer):
        self.widget.setLayer(layer)
        self.widget.labelCB.clear()
        self.widget.fillLabel()

    def linkageChanged(self, wrapper):
        self.setLinkage(wrapper.parameterValue())

    def setLinkage(self, linkage):
        self.widget.setLinkage(linkage)

    def value(self):
        return self.widget.value()

