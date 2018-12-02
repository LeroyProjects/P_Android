#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sarah Lachiheb et Cassandre Leroy
"""


from robosim import *
import math
import atexit
import main_genetic_tools as tools
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
    # Meilleurs fitness de tout les agents:
    bestFitness = 0.
    # Meilleurs parametre de tout les agents :
    bestParams = []
    # Parametre du robot parent pour 1+NB_ES :
    fitnessParent = 0.
    # Variables: 
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
        # Variable pour la fonction enregistrement de croisement :
        self.enregistrement = set()
        # Resevoir de parametre du au broadcast des autres robots :
        self.reservoir = dict() 
        
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
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-      Run     =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


    def step(self):
        """ Fonction run de l'agent applique a chaque pas d'iteration """
        self.stepSearchMethod()
        self.stepController()
        # Enregistrement des identifants des robots croisés lors du cycle d'évaluation :
        self.enregistrement_croisement()
        
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # =-=-=-=-=-=-=-=-=-=               Diffusion          -=-=-=-=-=-=-=-=-=-=-=-=-=
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        
    def enregistrement_croisement(self) :
        """ Mise a jour de l'ensemble des identifiants des robots a une distance de moins de maxSensorDistance dans un ensemble Set 
        
            => Il faut qu'il recupere l'ensemble des paramètres de ces robots en fin d'évaluation
        """ 
        # Recupere le sensor du Robot :
        sensorSelf = sensors[self.robot]        
        # Parcours l'ensemble des sensors :
        for index_sensor in range(len(SensorBelt)) :
            if sensorSelf[index_sensor].dist_from_border < maxSensorDistance and sensorSelf[index_sensor].layer == 'joueur' :
                # liste d'identifiant de robot qui ont croisés se robot :
                self.enregistrement.add(sensorSelf[index_sensor].sprite.numero)
        # Il faut que le génome de l'individu lui même soit dans le reservoir :
        self.enregistrement.add(self.id)
                
    def broadcast(self) :
        """ Fonction qui diffuse le paramtre du robot courant à l'ensemble des robots croisés durant son évaluation dans leurs parametre reservoir """ 
        # Parcours l'ensemble des identifiants de l'enregistrement :
        for index_robot in self.enregistrement :
            print("Broadcast :", index_robot)
            # Diffuse le parametre du robot courant vers ses voisins :
            agents[index_robot].setAddReservoir(self.id, self.params, self.fitness_fixe)
      
    def setAddReservoir(self, id, paramsVoisin, fitnessVoisin) :
        """ Un robot croisé durant le cycle d'évaluation lui ajoute son parametre dans son réservoir """
        self.reservoir[id] = (fitnessVoisin, paramsVoisin)    
        
    def listen(self) :
        """ Fonction qui recupere le paramtre du robot à l'ensemble des robots croisés durant son évaluation dans leurs parametre reservoir """
        # Parcours l'ensemble des identifiants de l'enregistrement :
        for index_robot in self.enregistrement :
            print("listen :", index_robot)
            # Met a jour le reservoir du robot courant :
            self.reservoir[index_robot] = (agents[index_robot].getFitness(),agents[index_robot].getParams())
            
    def format_reservoir(self) :
        """ Retourne le reservoir sous forme de liste (Dictionnaire => list) """
        list_params = list()
        # Parcours l'ensemble du réservoir :
        for item in self.reservoir.values():
            list_params.append((item[0], item[1]))
        return list_params
        
    def setEnregistrement(self, new_enregistrement):
        self.enregistrement = new_enregistrement
        
    def getEnregistrement (self) :
        return self.enregistrement
        
    def resetEnregistrement(self) :
        self.enregistrement = set()
        self.enregistrement.add(self.id)
        
    def getReservoir(self):
        return self.reservoir
        
    def resetReservoir(self) :
        """ Remise a zéro du reservoir """
        self.reservoir = dict()
                
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    # =-=-=-=-=-=-=-=-=-=               Fonction           -=-=-=-=-=-=-=-=-=-=-=-=-=
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    
    def initialisationGenome(self) :
        """ Initialise les parametre de l'agent avec une sequence de parametre aleatoire.
            Valeur des parametres aleatoire doit etre compris entre -10 et 10.
            Il y 18 parametres car 2*8 sensors + 2 biais.
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
        """ Modifie la fitness courante de l'agent """
        self.fitness_fixe = fit
    
    def setParams(self, newParams) :
        self.params = newParams

    def setTranslationValue(self,value):
        """ Normalisation de la translation avec application du movement de translation """
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
        """ Normalisation de la rotation avec application du movement de rotation """
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
        """ Mise a jour de la fitness de l'agent """
        self.fitness += math.fabs(translation) * (1-math.fabs(rotation)) * minSensorValue
    
    def resetFitness(self):
        """ Mise a zero de l'agent """
        self.fitness = 0
        
    def resetBestFitness() :
        """ Mise a zero de la meilleurs fitness de la classe Agent """
        Agent.bestFiness = 0
        
    def resetAgent(self) :
        """ Mise a zero des parametres de test de la classe Agent """
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


# '---------- Plot Box Plot ----------'

def moyenne_fitness_dans_generation(ens_generation, generator, nbAgents, nb_instance) :
    """ Retourne la moyenne de chaque generation sous forme de liste de liste """
    # Creation d'une liste de liste :
    moyenne_fitness_dans_generation = list()
        
    # Nombre de generation :
    for index_generation in range(generator) :
        p = list()
        # Nombre d'agent :
        for index_agent in range(nbAgents) :
            tmp = list()
            # Nombre d'instance : (Recupere la premier fitness de chaque generation)
            for index_instance in range(nb_instance) :
                tmp.append(ens_generation[index_instance][index_generation][index_agent])
            # Rajoute dans la liste :
            p.append(np.mean(tmp))
        moyenne_fitness_dans_generation.append(p)
    return moyenne_fitness_dans_generation
   
def meilleurs_fitness_par_instance(ens_generation, generator, nb_instance) :
    """ Retourne la meilleurs fitness pour chaque instance sous forme de liste """
    # Creation d'une liste de liste :
    meilleurs_fitness_par_instance = list()
    
    # Nombre de generation :
    for index_generation in range(generator) :
        liste = list()
        # Nombre d'instance :
        for index_instance in range(nb_instance) :
            # Recupere les generation par instance :
            liste.append((np.mean(ens_generation[index_instance][index_generation]), ens_generation[index_instance][index_generation]))
        # Recupere les meilleurs generation par instance :
        meilleurs_fitness_par_instance.append(sorted(liste, key=lambda x:x[0], reverse=True)[0][1])
    return meilleurs_fitness_par_instance



def create_box_plot(moyenne_fitness_dans_generation, meilleurs_fitness_par_instance, nom_box1, ylim1, nom_box2, ylim2, generator, nbAgents) :
    """ Creation de deux Box_plot :
            - Moyenne de chaque generation 
            - Meilleurs fitness pour chaque instance
    """
    plt.figure(figsize=(8, 3))
    plt.title('Moyenne sur 11 runs')
    plt.boxplot(moyenne_fitness_dans_generation)
    plt.xlabel('axes des y', fontsize = 14) 
    axes = plt.gca()
    axes.set_ylim(0, ylim1)
    axes.xaxis.set_ticklabels([generation * nbAgents for generation in range(1,generator+1)], fontsize = 8, verticalalignment = 'center')
    
    plt.xlabel('Evaluations')
    plt.ylabel('Performances')
    plt.savefig(nom_box1, bbox_inches='tight')
    plt.tight_layout()
    
    plt.figure(figsize=(8, 3))
    plt.title('Meilleur génération à chaque run')
    plt.boxplot(meilleurs_fitness_par_instance)
    plt.xlabel('axes des y', fontsize = 14) 
    axes = plt.gca()
    axes.set_ylim(0, ylim2)
    axes.xaxis.set_ticklabels([generation * nbAgents for generation in range(1,generator+1)], fontsize = 8, verticalalignment = 'center')
    
    plt.xlabel('Evaluations')
    plt.ylabel('Performances')
    plt.savefig(nom_box2, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

def ecrire_fichier_fitness(nom_fichier, meilleurs_parent) :
    """ Edite un fichier avec le meilleur individu de chaque instance avec ses parametres """
    with open(nom_fichier, "w") as f_write:
        for fit, param in (ens_fit_params) :
            f_write.write("Fitness : \n")
            f_write.write(str(meilleurs_parent[0]))
            f_write.write("\nParamettre : \n")
            f_write.write(", ".join(str(v) for v in meilleurs_parent[1])) 
            f_write.write("\n")
            
def ecrire_fichier_csv_fitness(nom_fichier, moyenne_fitness_dans_generation) : 
    """ Edite un fichier csv a partir des fitness de chaque generation """
    with open(nom_fichier, "w") as f_write:
        writer = csv.writer(f_write)
        for i,row in enumerate(moyenne_fitness_dans_generation):
            writer.writerow([i+1,(i+1*generator),row])



def main (iterator, generator, maxIteration, nb_instance, taux_mutation_elitiste, sigma, algorithme, str_selection, size_tournois, str_mutation, nbAgents, nb_enfants_par_parent_1_NB_ES) :
    """ Programme principal : Application de l'algorithme genetique
        Retoune l'ensemble des fitness pour chaque geneation, les meilleurs fitness et parametre de chaque instance  et le meilleur Agent à la fin de chaque instance"""
    
    # Ensemble des fitness pour multiples_generations generation
    ens_generation = list()
    # Recupere les meilleurs fitness et parametre de chaque instance :
    ens_fit_params = list()
    
    Agent.iterator = iterator
    
    ##########################
    init('empty',MyTurtle,screen_width,screen_height) 
    game.auto_refresh = False 
    game.frameskip = frameskip
    atexit.register(onExit)
    setupArena()
    setupAgents()
    game.mainiteration()
    ##########################
    
    for i in range (nb_instance) :
        
        # Affichage de la generation :
        print("Instance n°", i, " :\n")
        
        # Remise a zero de l'interation pour commencer une nouvelle generation :
        Agent.iteration = 0
        
        # Remise a zero de l'ensemble des fitness pour une generation :
        ens_fitness = list()
        
        while Agent.iteration != maxIteration:

            # Etape 1  =  Initialisation de chaque agent avec un genome aléatoire de type reel :
            if Agent.iteration  ==  0 :
                for i in range(nbAgents):
                    agents[i].resetAgent() 
                    agents[i].initialisationGenome()
            else :
                stepWorld()
                game.mainiteration()
                
                # Test de l'algorithme genetique :
                if (Agent.iteration % iterator == 0) : 

                    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                    # =-=-=-=-=-=-=-=-=-=         ELIMINATION DU BRUIT     -=-=-=-=-=-=-=-=-=-=-=-=-=
                    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                    
                    # Sauvegarde la constante iteration de la classe Agent :
                    save_iteration = Agent.iteration
                    
                    # Creation d'une liste de liste de fitness par agent en fonction des re-evaluations :
                    ens_evaluation_bruit = list()
                    ens_enregistrement = list()
                    for i  in range (nbAgents) :
                        ens_evaluation_bruit.append(list())
                        ens_enregistrement.append(list())
                
                    
                    # Recupere la fitness de chaque Agent / Recupere le reservoir de chaque agent :
                    for i in range(nbAgents) :
                        ens_evaluation_bruit[i].append(agents[i].getFitness())
                        ens_enregistrement[i].append(agents[i].getEnregistrement())
                        
                    
                    # Liste des moyennes des re-evaluations par agent :
                    ens_moyennne_agent_fitness = list()
                    
                    # Nombre de re-evaluation par agent qui doit etre effectue :
                    nb_revaluation_des_agents_avant_test = 4
                    
                    # Test d'evaluation pour faire la moyenne des fitness pour un type de parametre par agent:
                    for compteur in range(nb_revaluation_des_agents_avant_test) :
                        Agent.iteration = 1
                        # Execution des iterations pour un test :
                        for evaluate_sup in range(1,iterator+1) :
                            Agent.iteration += 1
                            
                            # Etape 2 = Evaluation de la population :
                            stepWorld()
                            game.mainiteration()
                            
                            # Test de l'algorithme genetique :
                            if (evaluate_sup % iterator == 0) : 
                                # Recupere l'ensemble des fitness et Recupere le reservoir de chaque agent de chaque generation de test :
                                for i in range(nbAgents) :
                                    ens_evaluation_bruit[i].append(agents[i].getFitness()) 
                                    ens_enregistrement[i].append(agents[i].getEnregistrement())
                                # Reset du reservoir à la fin d'une évaluation:
                                for i in range(nbAgents) :
                                    agents[i].resetEnregistrement()
    
                    # Faire la moyenne de chaque fitness d'agent :
                    for i in range(nbAgents) :
                        ens_moyennne_agent_fitness.append(np.mean(ens_evaluation_bruit[i]))
                        
                    # Mise a jour de la best fitness :
                    Agent.bestFitness = max(ens_moyennne_agent_fitness)
    
                    # Retablir la fitness des agents :
                    for i in range(nbAgents) :
                        agents[i].setFitness(ens_moyennne_agent_fitness[i])
                    
                    
                    # Retalir l'enregistrement de croisement de chaque agent :
                    for index_agent in range(nbAgents) :
                        moy_enregistrement = list()
                        for index_essais in range(nb_revaluation_des_agents_avant_test) :
                            moy_enregistrement.append((math.fabs(ens_evaluation_bruit[index_agent][index_essais] - ens_moyennne_agent_fitness[index_agent]), index_essais))
                        # Prendre la plus petite valeur :
                        select_enregistrement = sorted(moy_enregistrement, key=lambda x:x[0])[0]
                        # Retablir le reservoir de l'agent index_agent :
                        agents[i].setEnregistrement(ens_enregistrement[index_agent][select_enregistrement[1]])
                       
                    
             
                    # Remise a jour de la constante iteration apres la suppression des bruits:
                    Agent.iteration = save_iteration
                    
                    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                    # =-=-=-=-=-=-=-=-=-=         EVOLUTION DES AGENTS     -=-=-=-=-=-=-=-=-=-=-=-=-=
                    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                    
                    # ETAPE DE RECEPTION :
#                    
#                    for index_agent in range(nbAgents) :
#                        print("Part I :\n", agents[index_agent].getEnregistrement())
#                        print("Part I :\n", agents[index_agent].getReservoir())
                    
                    # Mise a jour des reservoirs pour chaque Agent :
                    for index_agent in range(nbAgents) :
                        # Diffusion
                        agents[index_agent ].broadcast()
                        # Ecoute
                        agents[index_agent ].listen()
  
                   
                    # ETAPE DE SELECTION et VARIATION :
                    for i in range(nbAgents) :              
                        # Creation d'une classe Genome_tools :
                        genome_tools = tools.Genome_tools(agents[i]) 
            
                        # Application de l'algorithme :
                        if algorithme  == 'partI'  :
                            # Algorithme genetique avec remplacement eliste  :
                            params_evolue = genome_tools.algorithme_genetique_eliste(str_selection, size_tournois, str_mutation, taux_mutation_elitiste, sigma)[0]

                        # Mise a jour :
                        agents[i].setParams(params_evolue)
                        agents[i].resetReservoir()
                        agents[i].resetEnregistrement()
                            
                        
                    # Recupere l'ensemble des fitness de chaque generation de test :
                    fit = [agents[i].getFitness() for i in range(nbAgents)]
                    ens_fitness.append(fit)
                    
            Agent.iteration +=  1


        # Ajoute la nouvelle generation dans la liste :
        ens_generation.append(ens_fitness) 
        
    return ens_generation




if __name__ == "__main__":
    
    """ 
    PRESENTATION DES VARIABLES POSSIBLES :
    ------------------------------------
    
    str_selection : 'selection_elitiste' / 'selection_roulette' / 'selection_par_rang' / 'selection_tournois' / 'selection_uniforme' / 'selection_echantillonnage_universel_stochastique'
               
    str_mutation : 'mutation_gaussienne' / 'mutation_uniforme_limite'
                
    size_tournois : a definir si utilisation de la selection par tournois, nombre de participant par tournois.
                
    taux_mutation : [0.0001, 0.1]
                    
    sigma :  [0.00000001, 0.01]
                
    algorithme : 'partI' /  '1_NB_ES' 
    """
    
    # Nombre d'iteration d'evolution des agents avant le test genetique :
    iterator = 100
    # Nombre de generation :
    generator = 2
    # Nombre iteration maximum :
    maxIteration = iterator * (generator + 1)
    # Nombre de generation a execute :
    nb_instance = 1
    
    # Choix d'algorithme :
    algorithme = 'partI'
    
    if algorithme == 'partI' :
        # Nombre d'agent au total :
        nbAgents = 2
        # Taux de mutation uniforme limite :
        taux_mutation_elitiste = 1
        # Sigma
        sigma = 0.01
        # Choix de selection :
        str_selection = 'selection_elitiste'
        # choix nombre de tournois :
        size_tournois = None
        # Choix de mutation :
        str_mutation = 'mutation_gaussienne'
        # Fonction main
        ens_generation = main (iterator, generator, maxIteration, nb_instance, taux_mutation_elitiste, sigma, algorithme, str_selection, size_tournois, str_mutation, nbAgents, None)
   
    
    # Recuperation des moyennes dans une generation :         
    moyenne_fitness_dans_generation = moyenne_fitness_dans_generation(ens_generation, generator, nbAgents, nb_instance)
    
    # Recuperation des meilleurs fitness par instance :
    meilleurs_fitness_par_instance = meilleurs_fitness_par_instance(ens_generation, generator, nb_instance)
    
    # Creation des Box plot :
    xlim_box1 = 800
    xlim_box2 = 1000
    nom_box1, nom_box2 = "Box_plot1.png", "Box_plot2.png"
    create_box_plot(moyenne_fitness_dans_generation, meilleurs_fitness_par_instance, nom_box1, xlim_box1, nom_box2, xlim_box2, generator, nbAgents)
    
    # Creation de la courbe moyenne :
    nom_fichier_1 = "donnees_courbe_moyenne.csv"
    ecrire_fichier_csv_fitness(nom_fichier_1, moyenne_fitness_dans_generation)
    













































