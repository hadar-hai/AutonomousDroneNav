import numpy as np
import pandas as pd
from shapely.geometry import Polygon
import matplotlib
matplotlib.use('Qt5Agg')


def preProcessObs(dfObstaclesRaw):
    # Preprocess dfObstacles
    dfObstacles = dfObstaclesRaw.rename({dfObstaclesRaw.columns[0]: 'NumPoints',
                                         dfObstaclesRaw.columns[1]: 'X',
                                         dfObstaclesRaw.columns[2]: 'Y',
                                         dfObstaclesRaw.columns[3]: 'Z',
                                         dfObstaclesRaw.columns[4]: 'obsType'}, axis=1)
    dfObstacles = dfObstacles.sort_values(by=['obsType'])
    return dfObstacles


# A Function to form a map from the obstacles data
def point2Obs(dfObstacles, pointMultiplier=4):

    # Set the polygons DF
    dfPoly = pd.DataFrame()
    dfPoly.insert(len(dfPoly.columns), column='Name', value=np.unique(dfObstacles.obsType))
    polyList = []
    xx = []
    yy = []
    for obs in dfPoly['Name']:
        dfObs = dfObstacles[dfObstacles.obsType == obs]
        pointList = []
        for point in range(len(dfObs)):
            pointList.append((dfObs.iloc[point].X, dfObs.iloc[point].Y))
        polyUpgraded = Polygon(pointList).convex_hull
        x, y = polyUpgraded.exterior.coords.xy
        xx = xx + x.tolist()
        yy = yy + y.tolist()
        polyList.append(polyUpgraded)
    dfPoly.insert(len(dfPoly.columns), column='Polygon', value=polyList)

    return dfPoly
