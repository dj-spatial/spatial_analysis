from qgis.core import QgsFeature, QgsPointXY, QgsGeometry
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.optimize import minimize

def getPointCoords(feat, weightFieldIndex):
    try:    
        pts = []
        weights = []
        for f in feat:
            pts.append(f.geometry().asPoint())
            if weightFieldIndex >=0:
                weights.append(f.attributes()[weightFieldIndex])
            else:
                weights.append(1)
        weights = np.asarray(weights, dtype = np.float32)

        x=[pt[0] for pt in pts]
        y=[pt[1] for pt in pts]
    except:
        pass
    else:
        return x, y, weights


def getMeanCenter(x, y, weights, id):
    try:
        wx = x * weights
        wy = y * weights
        mx=sum(wx)/sum(weights)
        my=sum(wy)/sum(weights)

        meanCenter = QgsPointXY(mx, my)

        centerFeat = QgsFeature()
        centerGeom = QgsGeometry.fromPointXY(meanCenter)
        attrs = centerFeat.attributes()
        centerFeat.setGeometry(centerGeom)
        attrs.extend([id])
        centerFeat.setAttributes(attrs)
    except:
        pass
    else:
        return centerFeat

def getMedianCenter(x, y, weights, id):
    try:
        sumWeights = sum(weights)
        mx=sum(x)/len(x)
        my=sum(y)/len(y)

        ## initial guesses
        cMedianCenter = np.zeros(2)
        cMedianCenter[0] = mx
        cMedianCenter[1] = my

        ## define objective
        def objective(cMedianCenter):
            return sum(np.sqrt((cMedianCenter[0] - x)**2 + (cMedianCenter[1] - y)**2)*weights/sumWeights)

        solution = minimize(objective, cMedianCenter, method='Nelder-Mead')
        medianCenter = QgsPointXY(solution.x[0], solution.x[1])
		
        centerFeat = QgsFeature()
        centerGeom = QgsGeometry.fromPointXY(medianCenter)
        attrs = centerFeat.attributes()
        centerFeat.setGeometry(centerGeom)
        attrs.extend([id])
        centerFeat.setAttributes(attrs)
    except:
        pass
    else:
        return centerFeat

def getCentralFeature(x, y, weights, id, dMetricIndex):
    try:
        dMetric = ['euclidean', 'cityblock']
        sumWeights = sum(weights)
        coords = np.stack([x, y], axis = -1)
        distanceVector = pdist(coords, metric = dMetric[dMetricIndex])
        distanceMatrix = squareform(distanceVector) / weights * sumWeights
        minDistanceIndex = np.argmin(sum(distanceMatrix))
        centralFeat = QgsFeature()
        centralGeom = QgsGeometry.fromPointXY(QgsPointXY(x[minDistanceIndex], y[minDistanceIndex]))
        attrs = centralFeat.attributes()
        centralFeat.setGeometry(centralGeom)
        attrs.extend([id])
        centralFeat.setAttributes(attrs)
    except:
        pass
    else:
        return centralFeat