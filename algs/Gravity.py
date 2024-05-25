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

import os

from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtGui import QIcon

from qgis.core import (QgsExpression,
                       QgsProcessingException,
                       QgsFeatureRequest,
                       QgsRasterFileWriter,
                       QgsProcessingContext,
                       QgsProcessingParameterExtent,
                       QgsProcessingParameterDistance,
                       QgsProcessingParameterCrs,
                       QgsField,
                       QgsFields,
                       QgsFeatureSink,
                       QgsProcessing,
                       QgsProcessingParameterField,
                       QgsProcessingParameterMatrix,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterRasterDestination)

from processing.algs.qgis.QgisAlgorithm import QgisAlgorithm
from ..utilities import getMeanCenter

import numpy as np
from osgeo import gdal
from osgeo import osr
import os.path
import re

pluginPath = os.path.split(os.path.split(os.path.dirname(__file__))[0])[0]


class Gravity(QgisAlgorithm):

    INPUT_POINTS = 'INPUT_POINTS'
    EXTENT = 'EXTENT'
    NROWS = 'NROWS'
    NCOLS = 'NCOLS'
    CRS = 'CRS'
    DISTANCE_FRICTION = 'DISTANCE_FRICTION'
    ATTRACTION_FIELDS = 'ATTRACTION_FIELDS'
    ATTRACTION_FACTORS = 'ATTRACTION_FACTORS'
    OUTPUT = 'OUTPUT'
	
    def icon(self):
        return QIcon(os.path.join(pluginPath, 'spatial_analysis', 'icons', 'gravity.svg'))

    def group(self):
        return self.tr('Gravity Model')

    def groupId(self):
        return 'gravitymodel'
    
    def name(self):
        return 'gravity'

    def displayName(self):
        return self.tr('Gravity')
    
    def msg(self, var):
        return "Type:"+str(type(var))+" repr: "+var.__str__()

    def __init__(self):
        super().__init__()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(self.INPUT_POINTS,
                                                              self.tr(u'Point Layer'),
                                                              [QgsProcessing.TypeVectorPoint]))
        self.addParameter(QgsProcessingParameterExtent(self.EXTENT, self.tr(u'Extent')))

        self.addParameter(QgsProcessingParameterNumber(self.NROWS,
                                                       self.tr(u'Rows'),
                                                       QgsProcessingParameterNumber.Integer,
                                                       100, False, 2, 1500))
        self.addParameter(QgsProcessingParameterNumber(self.NCOLS,
                                                       self.tr(u'Columns'),
                                                       QgsProcessingParameterNumber.Integer,
                                                       100, False, 2, 1500))
        self.addParameter(QgsProcessingParameterNumber(self.DISTANCE_FRICTION,
                                                       self.tr(u'Distance Friction'),
                                                       QgsProcessingParameterNumber.Double,
                                                       1.00, False, 0, 99999999))
        self.addParameter(QgsProcessingParameterField(self.ATTRACTION_FIELDS,
                                                      self.tr(u'Attraction Field'),
                                                      parentLayerParameterName=self.INPUT_POINTS,
                                                      type=QgsProcessingParameterField.Numeric,
                                                      allowMultiple = True))
        class ParameterAttraction(QgsProcessingParameterMatrix):
            def __init__(self, name='', description='', fields_param=None, default=None, optional=False):
                QgsProcessingParameterMatrix.__init__(self, name, description)
                self.fields_param = fields_param

            def clone(self):
                copy = ParameterAttraction(self.name(), self.description(), self.fields_param)
                return copy

        attraction_param = ParameterAttraction(self.ATTRACTION_FACTORS, self.tr(u'Attraction Factors'), fields_param=self.ATTRACTION_FIELDS)
        attraction_param.setMetadata({'widget_wrapper': {'class': 'spatial_analysis.forms.GravityAttraction.AttractionWidgetWrapper'}})
        self.addParameter(attraction_param)

        '''
        self.addParameter(QgsProcessingParameterMatrix(self.ATTRACTION_FACTORS,
                                                       self.tr(u'Attraction Factor(should be the same order of "Attraction Field")'),
                                                       numberRows = 1, 
                                                       headers=['Attraction Factor'],
                                                       defaultValue=[1]))
        '''
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT, self.tr('Gravity Raster')))

    def processAlgorithm(self, parameters, context, feedback):
        feedback.pushInfo(self.tr("Starting Algorithm: '{}'".format(self.displayName())))
        cLayer = self.parameterAsSource(parameters, self.INPUT_POINTS, context)
        nCols = self.parameterAsInt(parameters, self.NCOLS, context)
        nRows = self.parameterAsInt(parameters, self.NROWS, context)
        analysisCrs = context.project().crs()
        bbox = self.parameterAsExtent(parameters, self.EXTENT, context, analysisCrs)

        distanceFriction = self.parameterAsDouble(parameters, self.DISTANCE_FRICTION, context)
		
        attrNames = self.parameterAsFields(parameters, self.ATTRACTION_FIELDS, context)
        attrParams = self.parameterAsMatrix(parameters, self.ATTRACTION_FACTORS, context)
        if len(attrNames) is not len(attrParams):
            raise QgsProcessingException(
                self.tr('Attraction Field(s){0} should match Attraction Factor{1}').format(attrNames, attrParams))

        output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
        xcoords = np.empty((1, nCols))
        ycoords = np.empty((nRows, 1))

        #create a raster
        driver = gdal.GetDriverByName('GTiff')
        output_raster = driver.Create(output_path, xsize=nCols, ysize=nRows, bands=len(cLayer), eType=gdal.GDT_Float32)

        xres = (bbox.xMaximum()-bbox.xMinimum())/float(nCols)
        yres = (bbox.yMaximum()-bbox.yMinimum())/float(nRows)
        originX = [] 
        originY = [] 

        #create a grid
        feedback.pushInfo(self.tr(u'Creating a grid...'))
        feedback.pushInfo(self.tr(output_path))

        nxm = nCols + nRows
        for nCol in range(0, nCols):
            xcoords[:,nCol] = (bbox.xMinimum()+xres/2) + (nCol * xres)
            feedback.setProgress(int(nCols / nxm * 100))
        for nRow in range(0, nRows):
            ycoords[nRow,:] = (bbox.yMaximum()-yres/2) - (nRow * yres)
            feedback.setProgress(int((nRows + nCols) / nxm * 100))
        feedback.setProgress(0)

        #Cost Matrix
        feedback.pushInfo(self.tr(u'Creating a cost matrix...'))
        total = len(cLayer)
        costMatrix = []
        for i, f in enumerate(cLayer.getFeatures()):
            geom = f.geometry()
            originX.append(geom.asPoint().x())
            originY.append(geom.asPoint().y())
            costMatrix.append(np.sqrt((xcoords - originX[i])**2+(ycoords - originY[i])**2))
            feedback.setProgress(int(i / total * 100))
        feedback.setProgress(0)

        #Huff Calculation
        feedback.pushInfo(self.tr(u'Huff Calculation...'))
        huff = []
        for j, f in enumerate(cLayer.getFeatures()):
            attrs = 1
            for attrName, attrParam in zip(attrNames, attrParams):
                attr = pow(f[str(attrName)], float(attrParam))
                attrs = attrs * attr
            huff.append(attrs/(costMatrix[j]**distanceFriction))
            feedback.setProgress(int(j / total * 100))
        feedback.setProgress(0)

        #Huff Probability
        feedback.pushInfo(self.tr(u'Huff Probability...'))
        p_huff = []
        for k in range(0, len(cLayer)):
            p = huff[k]/sum(huff)
            p_huff.append(p) 
            output_raster.GetRasterBand(k+1).WriteArray(p)
            feedback.setProgress(int(k / total * 100))

        geotransform=(bbox.xMinimum(), xres, 0, bbox.yMaximum(),0, -yres)
        output_raster.SetGeoTransform(geotransform)  # Specify its coordinates
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(int(re.sub('EPSG:', '', analysisCrs.authid())))
        output_raster.SetProjection(srs.ExportToWkt())   # Exports the coordinate system 
        output_raster.FlushCache()
        output_raster = None

        results = {}
        results[self.OUTPUT] = output_path
        return results