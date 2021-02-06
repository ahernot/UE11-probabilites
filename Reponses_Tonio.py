#---Sujet 2020

#---------Préambule

#Chargement de dépendances

import numpy as np
import matplotlib.pyplot as plt

#Discrétisation
A=0
B=500

N=101 #Nombre de points de discrétisation
Delta = (B-A)/(N-1)
discretization_indexes = np.arange(N)
discretization = discretization_indexes*Delta

#Paramètres du modèle

mu=-5
a = 50
sigma2 = 12

#Données

observation_indexes = [0,20,40,60,80,100]
depth = np.array([0,-4,-12.8,-1,-6.5,0])

#Indices des composantes correspondant aux observations et aux componsantes non observées

unknown_indexes=list(set(discretization_indexes)-set(observation_indexes))


#---------Question 1

#Fonction C

def Covexp(dist,rangeval,sigmaval):
    return sigmaval * np.exp(-dist/rangeval)


#---------Question 2

distmat=abs(np.subtract.outer(discretization,discretization))


#---------Question 3

Sigma=Covexp(distmat,a,sigma2)


#---------Question 4

SigmaObs = Sigma[observation_indexes,:][:,observation_indexes]
SigmaObsUnknown = Sigma[observation_indexes,:][:,unknown_indexes]
SigmaUnknown = Sigma[unknown_indexes,:][:,unknown_indexes]


#---------Question 5

invSigma = np.linalg.inv(SigmaObs) 
Ec= mu+np.matmul(np.transpose(SigmaObsUnknown),np.matmul(np.linalg.inv(SigmaObs),depth-mu))

allval1 = np.zeros(N)
allval1[unknown_indexes]=Ec
allval1[observation_indexes]=depth
plt.plot(discretization,allval1)
plt.plot(discretization[observation_indexes], depth, 'ro')
plt.show()


#---------Question 6

SigmaCond = SigmaUnknown - np.matmul(np.transpose(SigmaObsUnknown),np.matmul(np.linalg.inv(SigmaObs),SigmaObsUnknown))

allval2 = np.zeros(N)
allval2[unknown_indexes]=np.diag(SigmaCond)
plt.plot(discretization,allval2)
plt.plot(discretization[observation_indexes], np.zeros(np.shape(observation_indexes)[0]), 'ro')
plt.show()


#---------Question 7

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


#---------Question 8

def length(z,delta):
    return sum(np.sqrt(Delta**2+(z[1:N]-z[0:-1])**2))


#---------Question 9

K=100
result = np.zeros(K)
for i in range(K):
    x=np.random.normal(0,1,np.shape(unknown_indexes)[0])
    allval3[unknown_indexes]=Ec + np.matmul(Cholesky,x)
    result[i]=length(allval3,Delta)

EspSimuCond = sum(result)/K
EspCond = length(allval1,Delta)


#---Sujet 2021


#---------Question 1

def simuCond(): 
    
    """
    Fonction de simulation conditionnelle pour pouvoir effectuer plusieurs générations.
    """

    global Cholesky, unknown_indexes, observation_indexes, Ec, depth
    x = np.random.normal(0,1,np.shape(unknown_indexes)[0])
    simu = Ec + np.matmul(Cholesky,x)

    allval3 = np.zeros(N)
    allval3[unknown_indexes]=simu
    allval3[observation_indexes]=depth

    return allval3

def simuCond_Juste(): 

    """
    Fonction de simulation conditionnelle pour pouvoir effectuer plusieurs générations 
    avec des valeurs de profondeur cohérentes.
    """
    global N
    allval3 = np.ones(N)*1 #Initialisation à l'état faux

    while (True):
        
        mask = allval3 <= 0 #Création d'un masque des profondeurs possibles (condition sur le tableau)

        if ( not(np.all(mask)) ): #S'il existe au moins une profondeur non valide

            allval3_correct = allval3*mask #Extraction profondeurs valides
            anti_mask = np.ones(np.shape(mask)) - mask #Masque avec les valeurs à changer
            allval3 = simuCond() #Génération d'une nouvelle série de points
            allval3 = allval3_correct + allval3*anti_mask #Remplacement des valeurs impossibles        

        else:
            break #Sortie de la boucle

    return allval3

#Test de fonctionnement du rejet (Validé)

plt.plot(discretization,simuCond_Juste())
plt.plot(discretization[observation_indexes], depth, 'ro')
plt.show()


#---------Question 2

#   Comparaison longueur moyenne de cable

K=1000
result_Juste = np.zeros(K)
for i in range(K):
    allval3 = simuCond_Juste()
    result_Juste[i]=length(allval3,Delta)

EspSimuCond_Juste = sum(result_Juste)/K

print(f'EspCondSimu = {EspSimuCond}')
print(f'EspCondSimu_Juste = {EspSimuCond_Juste}')

#On remarque une différence de l'ordre de quelques mètres peu importante. En effet, elle est souvent comprise
#dans l'intervalle de confiance calculé en partie I.

#   Comparaison histogramme des longueurs

plt.hist(result,50,density=True, label = "Simulation Partie I")
plt.hist(result_Juste,50,density=True, label = "Simulation avec rejet Partie II")
plt.legend(loc="best")
plt.show()

#On remarque que l'histogramme de la simulation avec rejet est plus "régulier". Il présente beaucoup moins
#de pics irréguliers qui étaient dûs aux altitudes positives éventuelles.

#   Comparaison des intervalles de confiance

print("Intervalle de confiance simulation Partie I : ", np.quantile(result,[0.025,0.975]))
print("Intervalle de confiance simulation avec rejet Partie II : ", np.quantile(result_Juste,[0.025,0.975]))

#COMMENTAIRE A FAIRE

#   Estimation du taux de rejet