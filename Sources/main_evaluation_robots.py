#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: Cassandre Leroy et Sarah Lachiheb
"""


from robosim import *
import math
import atexit
import random
import numpy as np
import csv
import matplotlib.pyplot as plt



'''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''
'''  variables globales   '''
'''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''

game = Game()

agents = []
screen_width=768
screen_height=768 



maxSensorDistance = 30             
maxRotationSpeed = 5
maxTranslationSpeed = 1

SensorBelt = [-170,-80,-40,-20,+20,40,80,+170]  # angles en degres des senseurs (ordre clockwise)

showSensors = True
frameskip = 0  
verbose = True


'''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''
'''  Fonctions init/step  '''
'''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''

def setupAgents():
    global screen_width, screen_height, nbAgents, agents, game

    # Make agents
    nbAgentsCreated = 0
    for i in range(nbAgents):
        while True:
            p = -1
            while p == -1: # p renvoi -1 s'il n'est pas possible de placer le robot ici (obstacle)
                p = game.add_players( (random.random()*screen_width , random.random()*screen_height) , None , tiled=False)
            if p:
                p.oriente( random.random()*360 )
                p.numero = nbAgentsCreated
                nbAgentsCreated = nbAgentsCreated + 1
                agents.append(Agent(p))
                break
    game.mainiteration()


def setupArena():
    for i in range(6,13):
        addObstacle(row=3,col=i)
    for i in range(3,10):
        addObstacle(row=12,col=i)
    addObstacle(row=4,col=12)
    addObstacle(row=5,col=12)
    addObstacle(row=6,col=12)
    addObstacle(row=11,col=3)
    addObstacle(row=10,col=3)
    addObstacle(row=9,col=3)

def updateSensors():
    global sensors
    # throw_rays...(...) : appel couteux (une fois par itération du simulateur). permet de mettre à jour le masque de collision pour tous les robots.
    sensors = throw_rays_for_many_players(game,game.layers['joueur'],SensorBelt,max_radius = maxSensorDistance+game.player.diametre_robot() , show_rays=showSensors)

def stepWorld():
    global sensors    
    updateSensors()
    # chaque agent se met à jour. L'ordre de mise à jour change à chaque fois (permet d'éviter des effets d'ordre).
    shuffledIndexes = [i for i in range(len(agents))]
    random.shuffle(shuffledIndexes)
    for i in range(len(agents)):
        agents[shuffledIndexes[i]].step()
    return


'''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''
'''  Fonctions internes   '''
'''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''

def addObstacle(row,col):
    # le sprite situe colone 13, ligne 0 sur le spritesheet
    game.add_new_sprite('obstacle',tileid=(0,13),xy=(col,row),tiled=True)

class MyTurtle(Turtle): # also: limit robot speed through this derived class
    maxRotationSpeed = maxRotationSpeed # 10, 10000, etc.
    def rotate(self,a):
        mx = MyTurtle.maxRotationSpeed
        Turtle.rotate(self, max(-mx,min(a,mx)))

def onExit():
    print ("\n[Terminated]")


'''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''
'''  Classe Agent/Robot   '''
'''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''

class Agent(object):
    
    # Compteur d'identification du robot :
    agentIdCounter = 0 
    # Identifiant du robot :
    id = -1
    # Référence du robot :
    robot = -1
    # Sigma Min :
    sigmaMin = 0.00000001
    # Sigma MAx : 
    sigmaMax = 0.01
    # BestFitness des l'agents :
    bestFitness = 0.
    # BestParam des Agents :
    bestParams = []
    # Parametre du robot rarent pour 1+NB_ES :
    fitnessParent = 0.
    sigma = 0.01
    # Variable
    iteration = 0
    iterator = 0

    def __init__(self,robot):
        # Incrémentation de la variable static :
        self.id = Agent.agentIdCounter
        Agent.agentIdCounter = Agent.agentIdCounter + 1
        # Instance de robot :
        self.robot = robot

        # Génome de l'agent :
        self.params = []
        # Fitness de l'agent :
        self.fitness = 0.
        # fitness courant :
        self.fitness_fixe = 0.0

        
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def stepSearchMethod(self): 
        """  Mise a jour du paramètre sigma de l'agent et de la fitness tout les n iterator :"""
        # Test si le temps de test est atteint :
        if Agent.iteration % Agent.iterator == 0:            
            # Met a jour la fitness du Robot parent :
            if self.id == 0 :
                Agent.fitnessParent = self.fitness

            # Met a jour la fitness du Robot enfant et parent :
            self.fitness_fixe = self.fitness
            
            # Met a jour la meilleurs fitness du programme :
            if Agent.bestFitness < self.fitness:
                Agent.bestFitness = self.fitness
                Agent.bestParams = self.params.copy()        
                
            # Affichage de la fitness/bestFitness et du numéro de l'évaluation de l'agent :
            
            if self.id == 0 :
                print ("Fitness Parent :",self.fitness)
                print ("Evaluation no.", int(Agent.iteration/Agent.iterator))
            else :
                print ("Fitness:",self.fitness)
                print ("Evaluation no.", int(Agent.iteration/Agent.iterator))
            
            # Repositionne les robots aux coordonnées initiales :
            p = self.robot
            p.set_position(screen_width/2,screen_height/2)
            p.oriente( random.random()*360 ) 
            
            # Remet la fitness à zéro :
            self.resetFitness()
            
            
    def evaluation_1_5_ES(self) :
        if Agent.fitnessParent < Agent.bestFitness :
            print("Enfant meilleurs\n")
            Agent.sigma = max(min(Agent.sigmaMax, self.sigma*2), Agent.sigmaMin)
        else :
            print("Parent meilleurs\n")
            Agent.sigma = max(2 ** (-1./4.) * self.sigma , Agent.sigmaMin)
        # Reset :
        Agent.bestFitness = 0

    def stepController(self):
        """ Fonction de paramètrage des moteurs de transaltion et de rotation via un réseau de neurone """
        # Initialisation des paramètres :
        p = self.robot
        sensor_infos = sensors[p]
        translation = 0
        rotation = 0
        k = 0
        
        # Réseau de neurone :
        minSensorValue = math.inf
        for i in range(len(SensorBelt)):
            tmp = sensor_infos[i].dist_from_border
            if tmp > 30 :
                tmp = 30
            if tmp < 0 :
                tmp = 0
            dist = tmp/maxSensorDistance
            if dist < minSensorValue :
                minSensorValue = dist
            translation += dist * self.params[k]
            k = k + 1
        
        translation += 1 * self.params[k]
        k = k + 1

        for i in range(len(SensorBelt)):
            tmp = sensor_infos[i].dist_from_border
            if tmp > 30 :
                tmp = 30
            if tmp < 0 : 
                tmp = 0
            dist = tmp/maxSensorDistance
            if dist < minSensorValue :
                minSensorValue = dist
            rotation += dist * self.params[k]
            k = k + 1

        rotation += 1 * self.params[k]
        k = k + 1
        
        # Borne la translation et la rotation :
        if translation > maxTranslationSpeed :
            translation = maxTranslationSpeed
        if rotation > maxRotationSpeed :
            rotation = maxRotationSpeed

        if translation < -maxTranslationSpeed :
            translation = -maxTranslationSpeed
        if rotation < -maxRotationSpeed :
            rotation = -maxRotationSpeed

        # Normalisation de la translation et de la rotation :
        if translation < 0:
            translation = (math.fabs(translation)) / (maxTranslationSpeed)
            translation = translation * (-1)
        else :
            translation = (math.fabs(translation)) / (maxTranslationSpeed)

        if rotation < 0:
            rotation = (math.fabs(rotation) ) / (maxRotationSpeed )
            rotation = rotation * (-1)
        else :
            rotation = (math.fabs(rotation) ) / (maxRotationSpeed )

        # Modification de la valeur de la rotation :
        self.setRotationValue(rotation)
        self.setTranslationValue(translation)

        # Mise a jour de la fitness :
        self.updateFitness(minSensorValue,translation, rotation)
        
        # Fin de fonction controller :
        return


    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


    def step(self):
        """ Fonction run de l'agent """
        self.stepSearchMethod()
        self.stepController()
     
    def initialisationGenome(self) :
        """ Initialise les parametre de l'agent avec une sequence de parametre aleatoire 
        
            Valeur des parametres aleatoire doit etre compris entre -10 et 10
            Il y 18 parametres car 2*8 sensors + 2 biais
        """
        self.params = [random.random()*10.0-5.0 for index in range(18)]

    def getRobot(self):
        """ Retourne une référence sur le robot """ 
        return self.robot
    
    def getSigma(self):
        """ Retourne la sigma de la loi de gausse """
        return self.sigma
  
    def getFitness(self) :
        """ Retourne la fitness de l'agent """
        return self.fitness_fixe
    
    def getBestFitness(self) :
        """ Retourne la meilleurs fitness de l'agent """
        return Agent.bestFitness
        
    def getParams(self) :
        """ Retourne le meilleurs parametre du genome de l'agent """
        return self.params
    
    def setFitness(self, fit) :
        self.fitness_fixe = fit
    
    def setParams(self, newParams) :
        self.params = newParams

    def setTranslationValue(self,value):
        if value > 1:
            print ("[WARNING] translation value not in [-1,+1]. Normalizing.")
            value = maxTranslationSpeed
        elif value < -1:
            print ("[WARNING] translation value not in [-1,+1]. Normalizing.")
            value = -maxTranslationSpeed
        else:
            value = value * maxTranslationSpeed
        self.robot.forward(math.fabs(value))

    def setRotationValue(self,value):
        if value > 1:
            print ("[WARNING] translation value not in [-1,+1]. Normalizing.")
            value = maxRotationSpeed
        elif value < -1:
            print ("[WARNING] translation value not in [-1,+1]. Normalizing.")
            value = -maxRotationSpeed
        else:
            value = value * maxRotationSpeed
        self.robot.rotate(value)

    def updateFitness(self, minSensorValue, translation, rotation):
        self.fitness += math.fabs(translation) * (1-math.fabs(rotation)) * minSensorValue
    
    def resetFitness(self):
        self.fitness = 0
        
    def resetBestFitness() :
        Agent.bestFiness = 0
        
    def resetAgent(self) :
        self.params = []
        self.fitness = 0
        self.fitness_fixe = 0
        Agent.bestFitness = 0
        Agent.bestParams = []

'''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''
'''  Main loop            '''
'''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''

def create_plot(ens_moyenne_fitness, generator) :
    plt.figure(figsize=(8, 3))
    plt.title("Détermination du nombre d'évaluations à faire pour supprimer le bruit")
    plt.plot([generation for generation in range(1,generator+1)], ens_moyenne_fitness)
    plt.xlabel("Nombre d'évaluations de la fitness pour un jeu de paramètre", fontsize = 10) 
    plt.ylabel("Fitness", fontsize = 10)
    plt.tight_layout()
    plt.show()
    
    """ Edite un fichier csv a partir des fitness de chaque generation """

    with open("test_evaluation_parametre_1.csv", "w") as f_write:
        writer = csv.writer(f_write)
        for i,row in enumerate(ens_moyenne_fitness):
            writer.writerow([i+1,i+1,row])
            

def main (iterator, generator, maxIteration) :
    
    
    Agent.iterator = iterator
    
    ##########################
    init('empty',MyTurtle,screen_width,screen_height) # display is re-dimensioned, turtle acts as a template to create new players/robots
    game.auto_refresh = False # display will be updated only if game.mainiteration() is called
    game.frameskip = frameskip
    atexit.register(onExit)
    setupArena()
    setupAgents()
    game.mainiteration()
    ##########################
    
    ens_fitness = list()
    ens_moyenne_fitness = list()
        
    # Iteration a l'infini :
    while Agent.iteration != maxIteration:

        # Etape 1  =  Initialisation de chaque agent avec un genome aléatoire de type reel :
        if Agent.iteration  ==  0 :
            # Test avec un Agent 1 :
            params = [3.3145948228192967, -3.2156775650216796, 5.155491462027963, 1.6800716347562097, 1.7661990045752929, 6.50569470669244, 4.791305980441474, 0.4932178817243549, 0.6398975099491113, 3.883282801126847, -5.154638978191092, -0.9643825335338349, -3.035973469105527, 4.094719387897268, -2.5154045895171344, 4.212147910430896, 1.920886289427297, -2.6324290135240247]
            # Test avec un Agent 2 :
            #params = [1, 0, 1, 1, 1, 1, -1, 0, 1, 0, -1, -1, -1, 0, 1, 1, 0, 1]
            for i in range(nbAgents) :
                agents[i].setParams(params)

        else :
            stepWorld()
            game.mainiteration()
                
            # Test de l'algorithme genetique :
            if (Agent.iteration % iterator == 0) : 
                    
                fitness = agents[0].getFitness()
                
                ens_fitness.append(fitness)
                    
                moy = np.mean(ens_fitness)
                    
                ens_moyenne_fitness.append(moy)
                    
        Agent.iteration +=  1
    
    return ens_moyenne_fitness




if __name__ == "__main__":

    
    # Nombre d'iteration d'evolution des agents avant le test genetique :
    iterator = 700
    # Nombre de generation :
    generator = 40
    # Nombre iteration maximum :
    maxIteration = iterator * (generator + 1)
    # Nombre de robot :
    nbAgents = 10
    

    # Fonction main :
    ens_fit_params = main (iterator, generator, maxIteration)
    
    # Affiche un courbe representant le nomdre de generation en fonction d'une fitness moyenne : 
    create_plot(ens_fit_params, generator)












































