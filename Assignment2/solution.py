####################################################################################################
###          CAP 4630 - Introduction to Artificial Intelligence                                  ###
###          Bryan Perdomo - Tyler Nguyen - Peterling Etienne                                    ###
###    Assignment 2: Traveling Sales Man Problem (TSA) using Genetic Algorithm  - 6/11/2021      ###
####################################################################################################
import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
'''“Given a list of cities and the distances between each pair of cities, 
what is the shortest possible route that visits each city and returns to the origin city?”'''
'''
Functions in this problem:
Gene: a city (represented as (x, y) coordinates)
Individual (aka “chromosome”): a single route satisfying the conditions above
Population: a collection of possible routes (i.e., collection of individuals)
Parents: two routes that are combined to create a new route
Mating pool: a collection of parents that are used to create our next population (thus creating the next generation of routes)
Fitness: a function that tells us how good each route is (in our case, how short the distance is)
Mutation: a way to introduce variation in our population by randomly swapping two cities in a route
Elitism: a way to carry the best individuals into the next generation
'''

#First step
'''
Create City Class
This class will allow us to create and handle our cities
It used to simply our x and y coordinates
'''
class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

'''
Create Fitness class
This class will do extra calculation to minimize route distance
The larger fitness score the better
'''
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0
    
    def routeDistance(self):
        if self.distance ==0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance
    
    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness

#Second step
'''
Create the population
'''