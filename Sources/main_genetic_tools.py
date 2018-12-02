#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: Cassandre Leroy et Sarah Lachiheb
"""

import random
import copy

class Genome_tools (object) :
    
    def __init__(self, agents, nbAgents):
        # Nombre d'agents dans la population :
        self.nbAgents = nbAgents
        # Recupere la liste d'agents :
        self.list_agents = agents
        # Mise en format de liste (fitness, agent) :
        self.list_population = [(agents[i].getFitness(), agents[i]) for i in range(nbAgents)]
        
    # 6 Modes de SELECTION  :
    def selection_elitiste(self, n) :
        """ Selectionne : les 'n' meilleurs fitness des agents parmis la population 
            Retourne    : les 'n' agents selectionnés
            
            n :             Nombre de parent à selectionne
        """
        # Recupere une population trie par la fitness et en recupere n:
        trie_select = sorted(self.list_population, key= lambda agent: agent[0], reverse=True)[:n]
        # Recupere seulement les agents :
        agents_select = [agent for fitness, agent in trie_select]
        return agents_select
    
    def selection_roulette(self, n):
        """ Selectionne : les 'n' selectionnés par la roulette wheel à partir d'un point fixe aléatoire.
            A chacun des individus de la population est associé un secteur d'une roue. 
            L'angle du secteur étant proportionnel à la fitness de l'individu qu"il représente.
            Retourne    : les 'n' agents selectionnés 
            
            n :             Nombre de parent à selectionne
            Rq : Ne fonctionne pas avec des fitness negatives car c'est une selection de proportion!!
        """
        
             
        # Selection des agents les plus prometteur par la roulette :
        parents_selectionnes = list()
        
        # Trie la liste de la population d'agent en fonction de leurs fitness :
        trie_listAgent = sorted(self.list_population, key=lambda agent: agent[0], reverse=True)
        
        somme_fitness = sum(self.list_population[ind][0]for ind in range(self.nbAgents))
        
        # Algorithme selection wheel :
        for index in range(n) :
            present = True
            while present :
                # Selectionne un point fixe de manière aléatoire sur la roue :
                point_fixe = random.random() * somme_fitness
                # Initialisation du point but :
                point_but = 0
                # Selection de l'agent le plus prometteur :
                for fitness, agent in trie_listAgent : #MODIF trie_listAgent
                    point_but += fitness
                    # L'individu pour lequel point_but dépasse point_fixe est l'agent choisi.
                    if point_but > point_fixe :
                        if agent in parents_selectionnes :
                            present = True 
                            continue
                        else :
                            present = False
                            parents_selectionnes.append(agent)
                            break
        return parents_selectionnes
    
    def selection_tournois(self, n, k_Tournois) :
        """ Selectionne les meilleurs agents parmis un tournois selectionnant les individus de maniere aleatoire
            Retourne : les 'n' agents selectionnés 
            
            n :             Nombre de parent à selectionne
            k_Tournois :    le nombre d'agents participants à chaque tournois 
        """
            
        # Initialisation  de la liste de selection des agents les plus prometteur par le tournois :
        parents_selectionnes = list()
        
        # Algorithme de selection tournois :
        for i in range(n) :
            # Initialisation des groupes de tournois :
            parents_tournois = list()
            # Selection des K agents qui participerons au tournois :
            for j in range(k_Tournois) :
                # Initialisation de la variable de present d'agent dans la liste parents_selectionnes :
                present = True
                while present :
                    # Selectionne de maniere aleatoire n agents différents :
                    agent_select = random.choice(self.list_population)
                    if agent_select in parents_selectionnes :
                        present = True
                    else :
                        parents_tournois.append(agent_select)
                        present = False
            # Selection du vainqueur du tournois :
            parents_selectionnes.append(max(parents_tournois, key= lambda agent: agent[0])[1])

            
        return parents_selectionnes
    
    def selection_uniforme(self, n) :
        """ La sélection se fait aléatoirement, uniformément et sans intervention de la valeur d'adaptation.
            Selectionne n agents de maniere aleatoire.
            Retourne les n agents selectionnes
            
            n :             Nombre de parent à selectionne
        """
        
        # Initialisation  de la liste de selection des agents :
        parents_selectionnes = list()
        
        # Algorithme de selection uniforme :
        for index in range(n) :
            # Initialisation de la variable de present d'agent dans la liste parents_selectionnes :
            present = True
            while present :
                # Selectionne de maniere aleatoire n agents différents :
                agent_select = random.choice(self.list_population)
                if agent_select in parents_selectionnes :
                    present = True
                else :
                    parents_selectionnes.append(agent_select[1])
                    present = False
                    
        return parents_selectionnes
    
    def selection_par_rang(self, n) :
        """ La selection par rang fonctionne egalement avec des valeurs de fitness negatives et surtout
            utilisee lorsque les agents de la population ont des valeurs de fitness très proches.
            Selection de la roue est proportionnel au rang dans la popultion est trie en fonction de la qualité
            de l'individu.
            Selectionne les n meilleurs agents.
            Retournes les n agents selectionnes
        
            n :             Nombre de parent à selectionne
        """
        
        # Initialisation  de la liste de selection des agents :
        parents_selectionnes = list()
        
        # Trie la population d'agent en fonction de la qualité de la fitness :
        trie_population = sorted(self.list_population, key=lambda agent:agent[0])
        
        # Attribuer un rang à chaque agent en fonction du trie de la population :
        rang_population = [(i+1, tuples[0], tuples[1]) for i, tuples in enumerate(trie_population)]
        
        # Somme des rang :
        somme_rang = sum(rang_population[i][0] for i in range(self.nbAgents))
        
        # Inversement de la liste de rang de population :
        rang_population.reverse()
        
        # Algorithme de selection par rang :
        for index in range(n) :
            present = True
            while present :
                # Selectionne un point fixe de manière aléatoire sur la roue :
                point_fixe = random.random() * somme_rang
                # Initialisation du point but :
                point_but = 0
                # Selection de l'agent le plus prometteur :
                for rang, fitness, agent in rang_population :
                    point_but += rang
                    # L'individu pour lequel point_but dépasse point_fixe est l'agent choisi.
                    if point_but > point_fixe :
                        if agent in parents_selectionnes :
                            present = True 
                        else :
                            present = False
                            parents_selectionnes.append(agent)
                        break
                            
        return parents_selectionnes
    
    def selection_echantillonnage_universel_stochastique(self, n) :
        """ L'echantillonnage universel stochastique est similaire à la selection à la roulette, mais
            au lieu d'avoir un point fixe, il en a plusieur. Par consequent, cela encourage les agents
            hautement aptes à êtres choisie au moins une fois.
            Selectionne les n meilleurs agents.
            Retournes les n agents selectionnes.
            
            n :             Nombre de parent à selectionne
            Rq : Ne fonctionne pas avec des fitness negatives car c'est une selection de proportion!!
        """
        
        # Initialisation  de la liste de selection des agents :
        parents_selectionnes = list()
        
        # Trie les agents de la population de maniere decroissante à leurs fitness :
        trie_listAgent = sorted(self.list_population, key=lambda agent:agent[0], reverse=True)
        
        # Somme les fitness de la population d'agent :
        sum_fitness = sum(self.list_population[ind][0] for ind in range(self.nbAgents))

        # Calcule la distance : les individus sont selectionnes par un ensemble de point equidistante :
        distance = sum_fitness / float(n) 
        
        # Selection d'un point de depart :
        start = random.uniform(0, distance)
        
        # Liste de point :
        points = [start + i*distance for i in range(n)]

        for p in points:
            i = 0
            sum_ = trie_listAgent[i][0]
            while sum_ < p:
                i += 1
                sum_ += trie_listAgent[i][0]
            parents_selectionnes.append(trie_listAgent[i][1])

        return parents_selectionnes
    
    
    # 1 Modes de croisement :
    
    def croisement(self, l_parent_select, n) :
        """ Croisement d'une population en utilisant un croisement en 1 point 
        
            l_population : liste correspondant à une population d'agent de classe Agent
            n : Nombre d'iteration
        """
        # Copie en profondeur de la l_population :
        copy_l_parent_select = copy.deepcopy(l_parent_select)
        # Liste d'enfants genere par le croisement de deux parents
        l_enfants = list()
        # Itere l'algorithme n fois pour creer 2 enfants a chaque iteration :
        for index in range(n) :
            # Selectionne le premier parent :
            parent1 = random.choice(copy_l_parent_select)
            copy_l_parent_select.remove(parent1)
            # Selectionne le deuxieme parent :
            parent2 = random.choice(copy_l_parent_select)
            copy_l_parent_select.remove(parent2)
            # Application du croisement en 1 point :
            enfant1, enfant2 = self.croisement_un_point(parent1, parent2)
            # Ajout des enfants dans la liste a retourner
            l_enfants.append(enfant1)
            l_enfants.append(enfant2)
        return l_enfants
    
    def croisement_un_point(self, parent1, parent2) :
        """ Un point de croisement aléatoire est selectionne et les queues de ses deux parents sont permutees
            pour obtenur deux enfants
            
            parent1 = Agent 1 : liste de parametre
            parent2 = Agent 2 : liste de parametre
            
            Retourne deux enfants
        """
        # On considere que les deux parents, on la même taille :
        taille_parent = len(parent1)
        
        # Selection aleatoire du point de croisement :
        point = random.randint(1, taille_parent - 1)
        
        # Algorithme de croisement :
        enfant1 = parent1[:point] + parent2[point:]
        enfant2 = parent2[:point] + parent1[point:]
        
        return enfant1, enfant2
        
    
    # 2 reel Modes de mutation :
    
    def mutation_gaussienne(self, parent_selectionnee, sigma, taux_mutation, nb_enfants_par_parent) :
        """ La fonction applique une mutation gaussienne selon un taux de mutation et du parametre
            sigma correspondant à l'evolution de la fitness de chaque agent.
            
            l_population : correspond à une liste de parametre
            taux_mutation : un taux de mutation à 1.
                        
            Rq : les parametres des genes sont bornes en -10, 10
        """
        # Liste d'enfant genere par les parents selectionnees :
        enfant_genere = list()
        # Algorithme de mutation uniforme limite :
        for parent in parent_selectionnee :
            # Genere nb_enfants_par_parent :
            for index in range(nb_enfants_par_parent) :
                # Generation d'un enfant identique au parent :
                enfant = copy.deepcopy(parent)
                # Test si on applique la mutation à cette agent :
                if random.random() < taux_mutation :
                    # Aleatoire gaussien :
                    random_gaussien = random.gauss(0, sigma)
                    # Application de la mutation gaussienne sur l'ensemble du genome de l'agent:
                    for i in range(len(enfant)) :
                        if enfant[i] + random_gaussien  > 10 :
                            enfant[i] = 10
                        elif enfant[i] + random_gaussien  < -10 :
                            enfant[i] = -10
                        else :
                            enfant[i] = enfant[i] + random_gaussien
                enfant_genere.append(enfant)
        # Retourne les enfants generes :
        return enfant_genere
    
    
    def mutation_uniforme_limite(self, parent_selectionnee, sigma, nb_enfants_par_parent) :
        """ La fonction applique une mutation uniforme 1/taille de population sur chaque gene de la population 
        
            l_population : correspond à une liste de parametre
            
            Rq : les parametres des genes sont bornes en -10, 10
        """
        # Liste d'enfant genere par les parents selectionnees :
        enfant_genere = list()
        # Algorithme de mutation uniforme limite :
        for parent in parent_selectionnee :
            # Genere nb_enfants_par_parent :
            for index in range(nb_enfants_par_parent) :
                # Generation d'un enfant identique au parent :
                enfant = copy.deepcopy(parent)
                # Parcours l'ensemble des genes de l'enfant:
                for gene in range(len(enfant)) :
                    # Applique une mutation de 1/population sur chaque gene du genome :
                    if random.random() < (1/self.nbAgents) : 
                        # Mutation uniforme :
                        enfant[gene] = random.uniform(-10, 10)
                # Ajoute le parent mute dans la liste des enfants generes :
                enfant_genere.append(enfant)
        # Retourne les enfants generes :
        return enfant_genere
                    
    
    # algorithme genetique avec un remplacement elitiste :
    def algorithme_genetique_eliste (self, str_selection, size_tounois, str_mutation, taux_mutation, sigma, nb_selection_parent) : 
        """ Le nombre parent selectionne doit être egale nbAgents // 2 
            Le taux de muation gaussienne doit être egale 1 : un parent donne un enfant.
        """
        # Nombre d'enfant par parent : 
        nb_enfants_par_parent = 1
        
        # Application de la selection sur l'ensemble de la population :
        if str_selection == 'selection_elitiste' :
            parents_selectionnes = self.selection_elitiste(nb_selection_parent)
        elif str_selection == 'selection_roulette' :
            parents_selectionnes = self.selection_roulette(nb_selection_parent)
        elif str_selection == 'selection_par_rang' :
            parents_selectionnes = self.selection_par_rang(nb_selection_parent)
        elif str_selection == 'selection_tournois' :
            parents_selectionnes = self.selection_tournois(nb_selection_parent,size_tounois)
        elif str_selection == 'selection_uniforme' :
            parents_selectionnes = self.selection_uniforme(nb_selection_parent)
        else :
            parents_selectionnes = self.selection_echantillonnage_universel_stochastique(nb_selection_parent)
                        
        # Recuperation des parametres des parents :
        parametres_parents = list()
        for agent in parents_selectionnes :
            parametres_parents.append(agent.getParams())

        # Creation d'enfant à partir de mutation d'enfant :
        if str_mutation == 'mutation_gaussienne' :
            enfant_genere = self.mutation_gaussienne(parametres_parents, sigma, taux_mutation, nb_enfants_par_parent) # Attention taux de mutation à 1.
        else :
            enfant_genere = self.mutation_uniforme_limite(parametres_parents, sigma, nb_enfants_par_parent)
            
        nouvelle_population = parametres_parents + enfant_genere
    
        return nouvelle_population
    
    
    def algorithme_1_nb_ES(self, sigma, nb_enfants_par_parent) :
        """ nb_selection_parent : Nombre de parent selectionne est à 1 
            str_selection : Utilisation possible => Elitiste et rien d'autre
            nb_enfants_par_parent : Nombre d'enfant genere par parent est de nb
            taux_mutation est toujours de 1
        """
        # Type de selection :
        str_selection = 'selection_elitiste'
        # Nombre de parent selectionnee :
        nb_selection_parent = 1
        # Type de mutation :
        str_mutation = 'mutation_uniforme_limite'
        
        # Selection d'un individu dans la population :
        if str_selection == 'selection_elitiste' :
            parents_selectionnes = self.selection_elitiste(nb_selection_parent)
            
        # Recuperation des parametres des parents :
        parametres_parents = list()
        for agent in parents_selectionnes :
            parametres_parents.append(agent.getParams())
            
        # Creation d'enfant à partir de mutation d'enfant :
        if str_mutation == 'mutation_uniforme_limite' :
            enfant_genere = self.mutation_uniforme_limite(parametres_parents, sigma, nb_enfants_par_parent)
            
        nouvelle_population = parametres_parents + enfant_genere

        return nouvelle_population    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
