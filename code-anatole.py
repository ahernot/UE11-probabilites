import numpy
from typing import Callable


def covarianceFunc(x: float or int, sigmaSquared: float or int = 12, a: float or int = 50):
    """
    Covariance estimation function
    :param x: point to evaluate the function at
    :param sigmaSquared: parameter
    :param a: parameter
    :return: value of the function at point x
    """
    value = sigmaSquared * numpy.exp( -1 * numpy.abs(x) / a)
    return value






# Available depth data
depthData = numpy.array([
    [0, 0],
    [20, -4],
    [40, -12.8],
    [60, -1],
    [80, -6.5],
    [100, 0]
])

# Requested values
requestedValues = numpy.linspace(0, 100, 200)


def getDepthArrayExperimental(depthData: numpy.ndarray, requestedValues: numpy.ndarray, covarianceFunc: Callable):
    """
    Calculate depth data for points at requestedValues, given depthData and covarianceFunc
    :param depthData: available depth data
    :param requestedValues: requested depths
    :param covarianceFunc: covariance function
    """


    """
    xPrev = requestedValues[0]

    for xVal in requestedValues:
        
        # Calculate distance to previous
        distToPrev = abs(xVal - xPrev)  # 2D distance

        # Calculate covariance with previous
        covariance = covarianceFunc(distToPrev)
    """


    # Calculate distance matrix
    distanceMatrix = numpy.abs(numpy.subtract.outer())
        