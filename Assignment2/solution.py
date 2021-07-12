####################################################################################################
###          CAP 4630 - Introduction to Artificial Intelligence                                  ###
###          Bryan Perdomo - Tyler Nguyen - Peterling Etienne                                    ###
###    Assignment 2: Traveling Sales Man Problem (TSA) using Genetic Algorithm  - 6/11/2021      ###
####################################################################################################
import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
'''“Travel Sales Man Problem : Given a list of cities and the distances between each pair of cities, 
what is the shortest possible route that visits each city and returns to the origin city?”'''
'''
Functions in this problem:
Gene: a city (represented as (x, y) coordinates)
Individual (aka “chromosome”): a single route satisfying the conditions
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
Init population by producing all the routes that satisfy our conditions
'''
def createRoute(cityList):
    #This function is selecting one random route from our list
    route = random.sample(cityList, len(cityList))
    return route

def initialPopulation(popSize, cityList):
    #This function will produce the list of the first population
    #By using above function to get the list of routes
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

#3rd step
'''
Determine fitness
Use Fitness class to rank each individual in the population
'''
def rankRoutes(population):
    fitnessResults = {}     #init a dictionary
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

#4th step
'''
Select the matching pool
This will create the next generation of routes
'''
def selection(popRanked, eliteSize):
    #This function will select the parents that will be used to create the next generation
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

def matingPool(population, selectionResults):
    #This function will use route IDs of the routes that will make up our mating pool from selection function
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

#5th step
'''
Breed
After mating pool created, next generation will be created in this process
'''
def breed(parent1, parent2):
    #This function will generate new generation from parents
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child


def breedPopulation(matingpool, eliteSize):
    #This function will generalize to create our offspring population
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

#6th step
'''
Mutation
Helps to avoid local convergence by introducing novel routes that will allow to explore other parts of the solution space
'''
def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            
            city1 = individual[swapped]
            city2 = individual[swapWith]
            
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

def mutatePopulation(population, mutationRate):
    mutatedPop = []
    
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

#7th step
def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


#GA
def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
    
    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute


# def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
#     pop = initialPopulation(popSize, population)
#     progress = []
#     progress.append(1 / rankRoutes(pop)[0][1])
    
#     for i in range(0, generations):
#         pop = nextGeneration(pop, eliteSize, mutationRate)
#         progress.append(1 / rankRoutes(pop)[0][1])
    
    
#     plt.plot(progress)
#     plt.ylabel('Distance')
#     plt.xlabel('Generation')
#     plt.show()
#Driver Code
if __name__ == "__main__":
    cityList = []

    for i in range(0,25):
        x=int(random.random() * 200)
        y=int(random.random() * 200)
        cityList.append(City(x, y))
        
    
    print("Param 1")
    geneticAlgorithm(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=100)
    # print("Param 2")
    # geneticAlgorithm(population=cityList, popSize=200, eliteSize=20, mutationRate=0.01, generations=200)
    # print("Param 3")
    # geneticAlgorithm(population=cityList, popSize=300, eliteSize=20, mutationRate=0.01, generations=100)
    # print("Param 4")
    # geneticAlgorithm(population=cityList, popSize=200, eliteSize=10, mutationRate=0.01, generations=300)
    # print("Param 5")
    # geneticAlgorithm(population=cityList, popSize=200, eliteSize=30, mutationRate=0.01, generations=100)

