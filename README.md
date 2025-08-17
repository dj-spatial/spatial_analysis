# SpatialAnalyzer
The Plugin implements spatial clustering, central tendancy and distribution to perform spatial analysis for Qgis. Users can choose analysis tools and the results will be displayed on the canvas.


## Features
1. Clustering(K-Means, Hierachical, DBScan)
2. Spatial Central Tendancy(Mean Center, Midian Center, Central Feature)
3. Spatial Distrubution(Standard Distance, Standard Deviation Ellipse)
4. Regression(Geographically Weighted Regression)
   - Works with point, line or polygon layers
   - Supports projected or spherical distances
   - Model choices: Gaussian for normal data, Poisson for counts, Logistic for binary data
   - Separate global and local explanatory variables with standardisation and variability tests
   - Optional forcing of global coefficients to their mean
   - Kernel options (fixed/adaptive Gaussian or bi-square) with bandwidth search and AICc/AIC/BIC criteria
   - Cross-validation bandwidth criterion available only for Gaussian model

## License
The SpatialAnalyzer Plugin is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation.

## Gallery
![SpatialAnalyzer](https://github.com/dj-spatial/spatial_analysis/assets/162799399/749b1c6d-940f-47b1-98b6-a9f2d8c39277)