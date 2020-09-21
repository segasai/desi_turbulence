import astropy.table as atpy
import scipy.spatial
import numpy as np


def getpoly(x, y, ndeg=2):
    # get the 2d polynomial design matrix
    polys = []
    for deg in range(1, ndeg + 1):
        for j in range(deg + 1):
            i1, i2 = j, deg - j
            polys.append(
                np.concatenate((i1 * x**(i1 - 1 + (i1 == 0)) * y**i2,
                                i2 * x**i1 * y**(i2 - 1 + (i2 == 0)))))
    polys = np.array(polys)

    return polys


def predictor(curx, cury, curdx, curdy, tofit, ndeg=2, polys=None):
    #  fit the offsets by a polynomial f-n
    # return the model predictions and
    # the cross-validated norm
    ncv = 3
    if polys is None:
        polys = getpoly(curx, cury, ndeg=ndeg)
    else:
        polys = polys[:(ndeg * (ndeg + 1)) // 2 - 1]
    cvid = np.arange(2 * len(curx)) % ncv
    norm = 0
    dxy = np.concatenate((curdx, curdy))
    tofit2 = np.concatenate((tofit, tofit))
    for i in range(ncv):
        cursub = tofit2 & (cvid != i)
        cursub1 = tofit2 & (cvid == i)
        xcoeff = scipy.linalg.basic.lstsq(polys[:, cursub].T,
                                          dxy[cursub],
                                          check_finite=False)[0]
        xpred = np.dot(polys.T, xcoeff)[cursub1]
        norm += np.sum((xpred - dxy[cursub1])**2)
    xcoeff = scipy.linalg.basic.lstsq(polys[:, tofit2].T, dxy[tofit2])[0]
    xpred, ypred = np.dot(polys.T, xcoeff)[~tofit2]
    return xpred, ypred, norm, polys


def correcter(x, y, x0, y0, win=50):
    """
    Parameters
    ----------
    x: ndarray
        Measured x
    y: ndarray
        Measured y
    x0: ndarray
        Expected x
    y0: ndarray
        Expected y

    Returns
    -------
    xy: tuple of ndarray
        Tuple of corrected arrays
    
    """
    maxndeg = 4
    # offsets wrt reference
    dx = x - x0
    dy = y - y0

    X0 = np.array([x0, y0]).T
    X = np.array([x, y]).T

    T0 = scipy.spatial.cKDTree(X0)
    N = len(x)

    # predicted offset
    dxpred = np.zeros(N)
    dypred = np.zeros(N)
    bestdegs = np.zeros(N)

    for i in range(N):
        # go over each point
        # query the neighborhood
        xids = T0.query_ball_point(X[i], win)
        xids = np.array(xids)

        curxcen, curycen = x[i], y[i]
        curx = x[xids] - curxcen
        cury = y[xids] - curycen
        tofit = xids != i
        curdx = dx[xids]
        curdy = dy[xids]
        bestnorm = 1e9
        # try polynomials of different degrees
        # selecting by cross-validate norm
        polys = None
        for ndeg in range(maxndeg, 0, -1):
            xpred, ypred, norm, polys = predictor(curx,
                                                  cury,
                                                  curdx,
                                                  curdy,
                                                  tofit,
                                                  ndeg=ndeg,
                                                  polys=polys)
            if norm < bestnorm:
                bestnorm = norm
                lastx, lasty = xpred, ypred
                bestdeg = ndeg
        dxpred[i] = lastx
        dypred[i] = lasty
        bestdegs[i] = bestdeg
    return x - dxpred, y - dypred


def doit_file(input_csv, expected_csv, output_csv, win=50):
    """
    Parameters: 
    input_csv
    expected_csv
    output_csv
    win -- window size to use to fit for local turb
    """
    tab = atpy.Table().read(input_csv)
    tab['XNEW'] = tab['X_FP'] + np.nan
    tab['YNEW'] = tab['X_FP'] + np.nan
    tab['DEG'] = tab['X_FP'] + np.nan

    tab0 = atpy.Table().read(expected_csv)
    maxap = 0.05
    # maximum aperture to x-match expected vs observed

    # reference positions
    X0 = np.array([tab0['X_FP'], tab0['Y_FP']])

    # observed positions
    X = np.array([tab['X_FP'], tab['Y_FP']])

    dist, xid = scipy.spatial.cKDTree(X0.T).query(X.T)
    xset = dist < maxap
    xset0 = xid[xset]

    x, y = tab['X_FP'][xset], tab['Y_FP'][xset]
    x0, y0 = tab0['X_FP'][xset0], tab0['Y_FP'][xset0]
    x1, y1 = correcter(x, y, x0, y0)
    # corrected coordinates
    tab['XNEW'][xset] = x1
    tab['YNEW'][xset] = y1

    # these are just from expected.csv
    tab['XDUMB'] = np.zeros(len(tab['X_FP'])) + np.nan
    tab['YDUMB'] = np.zeros(len(tab['X_FP'])) + np.nan
    tab['XDUMB'][xset] = x0
    tab['YDUMB'][xset] = y0

    tab.write(output_csv, overwrite=True)
