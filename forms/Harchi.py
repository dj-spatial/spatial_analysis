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
from scipy.cluster.vq import whiten
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram
import plotly
import plotly as plt
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from qgis.PyQt import uic
from qgis.core import QgsNetworkAccessManager, QgsProject, QgsMessageLog, QgsProcessingUtils
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
        self.chart_id = 0
        self.cluster_method = 0

        # Connect signals
        self.plotBtn.clicked.connect(self.plotView)
        self.browserBtn.clicked.connect(self.browserView)
        self.chartType.buttonClicked.connect(self.chart_type)
        self.clusterBy.buttonClicked.connect(self.cluster_by)

    def setSource(self, source):
        if not source:
            return
        self.source = source

    def setOptions(self, options):
        self.options = options
        
    def setLinkage(self, linkage):
        self.linkageIdx = linkage

    def chart_type(self):
        self.labelCB.setEnabled(False)
        if self.dendrogramRB.isChecked():
            self.labelCB.setEnabled(True)
            self.chart_id = 0 
        elif self.correlationRB.isChecked():
            self.chart_id = 1
        elif self.distanceRB.isChecked():
            self.chart_id = 2

    def fillLabel(self):
        cLayer =  self.source
        if cLayer is None:
            msg = u'No Layer Selected.'
            return msg
        fields = cLayer.fields()
        self.fNames = fields.names()
        self.labelCB.addItems(self.fNames)

    def cluster_by(self):
        self.cluster_num.setEnabled(False)
        self.cophenet.setEnabled(False)
        if self.byNum.isChecked():
            self.cluster_method = 0
            self.cluster_num.setEnabled(True)
        elif self.cutTree.isChecked():
            self.cluster_method = 1
            self.cophenet.setEnabled(True)

    def getHarchi(self):
        cLayer = self.source
        if cLayer is None:
            msg = u'No Layer Selected.'
            return msg
        to_cluster, variable_fields, normalized = self.options

        # input --> numpy array
        if to_cluster == 'geom':
            features = [[f.geometry().centroid().asPoint().x(), f.geometry().centroid().asPoint().y()] for f in cLayer.getFeatures()]
            features = np.stack(features, axis = 0)
        else:
            features = [[f[vf] for f in cLayer.getFeatures()] for vf in variable_fields]
            features = np.stack(features, axis = 1)

        if normalized:
            features = whiten(features)

        dMethod = ['centroid', 'ward', 'single', 'complete', 'average']
        if self.chart_id == 0:
            Z = linkage(features, dMethod[self.linkageIdx], metric = 'euclidean')
            ddata = dendrogram(Z)
            
            labels=[f.attributes()[self.labelCB.currentIndex()] for f in cLayer.getFeatures()]
            hText=list(Z[:,2])
            dendro = ff.create_dendrogram(features, linkagefun = lambda x : Z, orientation = 'bottom', labels = labels, hovertext = ddata['dcoord'])

            #그래프
            dendro['layout'].update({'autosize': True, 'plot_bgcolor': 'white', 'title' : 'Dendrogram by ' + dMethod[self.linkageIdx] + ' method' })
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
        elif self.chart_id == 1:
            #Cophenet Correlation
            cophenet_corr = []
            for m in dMethod:
                Z = linkage(features, method = m, metric = 'euclidean')
                c,_ = cophenet(Z, pdist(features))
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
            Z = linkage(features, method = dMethod[self.linkageIdx], metric = 'euclidean')
            cDistance = Z[:,2]
            diff = np.diff(cDistance, 1)
            revDistance = cDistance[::-1]
            revDiff = np.insert(diff[::-1], 0, 0, axis=0)
            x_range = np.arange(1, len(cDistance) + 1)
            dHover = [u'Last ' + str(n) + u'th Cluster' + '<br>' + 'Cophenet Distance = ' + '{0:,.1f}'.format(d) for n, d in zip(x_range, revDistance)]
            diffHover = [u'Last ' + str(n) + u'th Cluster' + '<br>' + u'Distance Decline = ' + '{0:,.1f}'.format(d) for n, d in zip(x_range, revDiff)]

            trace0 = go.Bar(
                x=x_range,
                y=revDiff,
                marker=dict(color='white', line=dict(width=1, color='navy')),
                text=diffHover,
                hoverinfo='text',
                name='Distance Decline'
            )
            trace1 = go.Scatter(
                x=x_range,
                y=revDistance,
                yaxis='y2',
                text=dHover,
                mode="lines",
                line=dict(color='orange'),
                hoverinfo='text',
                hoverlabel=dict(bgcolor='white'),
                textposition='top center',
                name=u'Cophenet Distance'
            )
            x_axis = {
                'title': u'Last Nth Cluster Merge',
                'dtick': 5,
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
                'title': u'Distance Decline',
                'anchor': 'x',
                'ticks': '',
                'color': 'black',
                'zeroline': True,
                'showline': True,
                'showgrid': True,
                'autorange': True,
                'rangemode': 'normal',
                'visible': True,
                'mirror':False
            }
            y_axis2 = {
                'type': '-',
                'title': u'Cophent Distance',
                'anchor': 'x',                
                'side': 'right',
                'overlaying': 'y',
                'ticks': '',
                'zeroline': True,
                'showline': True,
                'showgrid': False,
                'autorange': True,
                'rangemode': 'normal',
                'visible': True,
                'mirror':False
            }
            layout = {
                'yaxis': y_axis,
                'yaxis2': y_axis2,
                'plot_bgcolor': 'white',
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
        
    def setValue(self, value):
        return True

    def value(self):
        if self.cluster_method == 0:
            threshold = self.cluster_num.value()
        else:
            threshold = self.cophenet.value()
        return [self.cluster_method, threshold]

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
                self.setSource(wrapper.parameterValue())
                wrapper.widgetValueHasChanged.connect(self.layerChanged)
            elif wrapper.parameterDefinition().name() == self.param.variable_options:
                self.setOptions(wrapper.parameterValue())
                wrapper.widgetValueHasChanged.connect(self.optionsChanged)
            elif wrapper.parameterDefinition().name() == self.param.linkage_param:
                self.setLinkage(wrapper.parameterValue())
                wrapper.widgetValueHasChanged.connect(self.linkageChanged)

    def layerChanged(self, wrapper):
        self.setSource(wrapper.parameterValue())

    def setSource(self, source):
        source = QgsProcessingUtils.variantToSource(source, self.context)
        self.widget.setSource(source)
        self.widget.labelCB.clear()
        self.widget.fillLabel()

    def optionsChanged(self, wrapper):
        self.setOptions(wrapper.parameterValue())

    def setOptions(self, options):
        self.widget.setOptions(options)

    def linkageChanged(self, wrapper):
        self.setLinkage(wrapper.parameterValue())

    def setLinkage(self, linkage):
        self.widget.setLinkage(linkage)

    def setValue(self, value):
        return self.widget.setValue(value)

    def value(self):
        return self.widget.value()

