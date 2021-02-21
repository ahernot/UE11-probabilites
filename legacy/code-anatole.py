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
        









# Chargement de dépendances
import numpy as np
import matplotlib.pyplot as plt

# Discrétisation
A=0
B=500
N=101  # nombre de points de discrétisation
Delta = (B-A)/(N-1)
discretization_indexes = np.arange(N)
discretization = discretization_indexes*Delta

# Paramètres du modèle
mu=-5
a = 50
sigma2 = 12

# Données
observation_indexes = [0,20,40,60,80,100]
depth = np.array([0,-4,-12.8,-1,-6.5,0])

# Indices des composantes correspondant aux observations et aux componsantes non observées
unknown_indexes=list(set(discretization_indexes)-set(observation_indexes))



# Fonction C
def Covexp(dist,rangeval,sigmaval):
    return sigmaval * np.exp(-dist/rangeval)


# Matrice de distance
distmat=abs(np.subtract.outer(discretization,discretization))


# Matrice de covariance de Z
Sigma=Covexp(distmat,a,sigma2)


# Matrices de covariance
SigmaObs = Sigma[observation_indexes,:][:,observation_indexes]  # entre les observations
SigmaObsUnknown = Sigma[observation_indexes,:][:,unknown_indexes]  # entre les observations et les inconnues
SigmaUnknown = Sigma[unknown_indexes,:][:,unknown_indexes]  # entre les inconnues



# Espérance conditionnelle
invSigma = np.linalg.inv(SigmaObs) 
Ec= mu+np.matmul(np.transpose(SigmaObsUnknown),np.matmul(np.linalg.inv(SigmaObs),depth-mu))

allval1 = np.zeros(N)
allval1[unknown_indexes]=Ec
allval1[observation_indexes]=depth
plt.plot(discretization,allval1)
plt.plot(discretization[observation_indexes], depth, 'ro')
plt.show()




# Matrice de variance conditionnelle
SigmaCond = SigmaUnknown - np.matmul(np.transpose(SigmaObsUnknown),np.matmul(np.linalg.inv(SigmaObs),SigmaObsUnknown))

allval2 = np.zeros(N)
allval2[unknown_indexes]=np.diag(SigmaCond)
plt.plot(discretization,allval2)
plt.plot(discretization[observation_indexes], np.zeros(np.shape(observation_indexes)[0]), 'ro')
plt.show()



# Simulation conditionnelle
Cholesky = np.linalg.cholesky(SigmaCond)
x = np.random.normal(0,1,np.shape(unknown_indexes)[0])
simu = Ec + np.matmul(Cholesky,x)

allval3 = np.zeros(N)
allval3[unknown_indexes]=simu
allval3[observation_indexes]=depth
plt.plot(discretization,allval3)
plt.plot(discretization,allval1)
plt.plot(discretization[observation_indexes], depth, 'ro')
plt.show()



# Longueur du câble
def length(z,delta):
    return sum(np.sqrt(Delta**2+(z[1:N]-z[0:-1])**2))

K=100000
result = np.zeros(K)
for i in range(K):
    x=np.random.normal(0,1,np.shape(unknown_indexes)[0])
    allval3[unknown_indexes]=Ec + np.matmul(Cholesky,x)
    result[i]=length(allval3,Delta)
sum(result)/K

length(allval1,Delta)



# Suite Mn
indice_simu = 1+np.arange(K)
plt.plot(indice_simu,np.cumsum(result)/indice_simu)
plt.show()


# Histogramme des longueurs de câble
plt.hist(result,50,density=True)
plt.show()


# Intervalle de confiance
Ln = sum(result)/K
sigman = np.std(result)
[Ln - sigman*1.96,Ln + sigman*1.96]
np.quantile(result,[0.025,0.975])  # méthode 2


# Proba > 525m
np.mean(result>525)
