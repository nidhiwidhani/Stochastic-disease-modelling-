## TODO:
# implement bounding box ----DONE
# exponential decay of infection radius ---DONE with bugs
# sampling --- DONE
# recovering people -- DONE with bugs
# removed population -- DONE with bugs
# population centers --\___ use potential mapping MAYBE?
# social distancing  --/

import random
from random import random as rand
from matplotlib import pyplot as plt
import time
from math import sqrt
from math import e
from math import pi
"""
parameters for the simulation
"""
""" ---------------------------------------------"""
# total population
total = 100


# population susceptibility parameters,expressed as a fraction of a whole
initial_infected_prob = 0.01

susceptible_frac = 1
infected_frac = 0.0
deceased_frac = 0.0
recovered_frac = 0.0

susceptible = susceptible_frac * total
infected = infected_frac * total
deceased = deceased_frac * total
recovered = recovered_frac * total

# simulation characterstics
city_size = {"length":1,"width":1}

total_steps = 50 
time_step = 0.001
time_steps_in_day = 6

avg_velocity = 0.02 # avg distance to be moved by a person in a time step.

# disease characterstics
prob_infection = 0.8
prob_recovery = 0.85
prob_death = 1 - prob_recovery

max_time_to_recovery_days = 12
sickest_time = 7 # infection peak
std_sickness = 3 # standard deviation of duration of symptoms

rad_infection = 0.1

normalising_factor = 0.25

# population aggragates
population_stats = []
population_aggregate = []


# testing characterstics
day_of_test = int(total_steps * 0.5)
num_samples = int(total * 0.2)

# test characterstics
sensitivity = 0.9
specificity = 0.9

""" ---------------------------------------------"""
"""
Miscellaneous functions for the simulation including
1)  distance
    takes two people as parameter
    Distance between two people ( Normal L2 Norm)

2) plot_aggregate
    Using SIR model,    "green" indicates susceptible
                        "red"   indicates infected
                        "blue"  indicates recovered
                        "black" indicates deceased
3) sample
    takes a day to sample and the number of people to sample as parameters
    Random sampling among alive people has been chosen.
    The sampling characterstics can be changed in the parameters above.
    default is 20% population.
    The day of sampling has been chosen right in between

4) test_sample
    Takes a sample_instance as parameter
    Test has a given sensitivity and specificity as chosen in parameters above.
    default is 90% for both.

5) simulate
    To start the simulation

6) test_results


"""
""" ---------------------------------------------"""

def distance(person_i,person_j):
    i_x = person_i.position["x"]
    i_y = person_i.position["y"]
    j_x = person_j.position["x"]
    j_y = person_j.position["y"]

    return sqrt((abs(i_x - j_x))**2 + (abs(i_y - j_y))**2) # L2 Norm

def plot_aggregate():
    S = [i["susceptible"] for i in population_stats]
    I = [i["infected"] for i in population_stats]
    R = [i["recovered"] for i in population_stats]
    D = [i["deceased"] for i in population_stats]

    final = plt.figure(3)
    plt.bar(range(len(S)),S,color = ['g'])
    plt.bar(range(len(I)),I,color = ['r'])
    plt.bar(range(len(R)),R,color = ['b'])
    plt.bar(range(len(D)),D,color = ['k'])

    plt.show(block=False)
    plt.pause(20) # Change time here for the duration of chart
    plt.close()

def sample(day,num):
    alive_population = list(filter(lambda x: x.status == True,population_aggregate[day]))
    return random.sample(alive_population,num)

def test_sample(sample_inst):

    for i in sample_inst:
        if i.infected == True:
            test_result = True if rand() < sensitivity else False
            i.update_test_result(test_result)
        elif i.infected == False:
            test_result = False if rand() < specificity else True
            i.update_test_result(test_result)
    test_final_results = [{"test":i.test_result,"actual":i.infected} for i in sample_inst]
    return(test_final_results)


def simulate():
    city_inst = city()
    population_inst = population(city_inst)

    for i in range(total_steps):
        population_aggregate.append(population_inst.next_step())
        population_inst.plot_position()
        population_stats.append(population_inst.plot_stats())

def test_results():

    TP = 0 # True Positives
    TN = 0 # True Negatives
    FP = 0 # False Positives
    FN = 0 # False Negatives

    for i in results:
        if i["test"] == True and i["actual"] == True:
            TP += 1
        if i["test"] == False and i["actual"] == False:
            TN += 1
        if i["test"] == True and i["actual"] == False:
            FP += 1
        if i["test"] == False and i["actual"] == True:
            FN += 1

    total_tests = TP + TN + FP + FN

    accuracy = (TP + TN)/total_tests
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    f1_score = 2/((1/recall) + (1/precision)) # Straight definitions

    print(results)# ====================
    print("TP = {}".format(TP))
    print("TN = {}".format(TN))
    print("FP = {}".format(FP))
    print("FN = {}".format(FN))
    print("total = {}".format(total_tests))
    print("-------------")
    print(" accuracy = {}".format(accuracy))
    print(" precision = {}".format(precision))
    print(" recall = {}".format(recall))
    print(" f1_score = {}".format(f1_score))
""" ---------------------------------------------"""
"""
Class definition of city.
Not much to talk about here, just a boudning box for the city.
neighboring samples have been chosen to bound the population
"""
""" ---------------------------------------------"""

class city:
    def __init__(self):
        self.city_size = city_size
        self.l = city_size["length"]
        self.w = city_size["width"]

""" ---------------------------------------------"""
"""
Class definition of a person
A person has a randomly generated starting poisiton and an initial infection status.
A person also has the following characterstics

number : For tracking
position : for position
status : True for Alive, False for dead
infected : True for infected, False for not infected
recovered : True if a person has had the disease and recovered, False otherwise
time_since_infection : Time steps since infection. To calculate mortality
test_result : result of the test taken by the person

The class also has dedicated methods to update said characterstics with the template update_(insertattribute here)
"""
""" ---------------------------------------------"""

class person:
    def __init__(self,city_inst,number):
        self.number = number
        self.position = {"x":rand() * city_inst.l,"y":rand() * city_inst.w}
        self.infected = True if rand() <= initial_infected_prob else False
        self.status = True
        self.recovered = False
        self.time_since_infection = -1
        self.test_result = False

    def update_position(self,new_position):
        self.position = new_position

    def update_status(self,new_status):
        self.status = new_status

    def update_infected(self,new_infected):
        self.infected = new_infected

    def update_recovered(self,new_recovered):
        self.recovered = new_recovered

    def update_test_result(self,new_test_result):
        self.test_result = new_test_result
""" ---------------------------------------------"""
"""
Class definition of population
A population has following attributes
total : Total number of people in the population
population :    A live snapshot of the population as is. The records are maintained
                in population_stats and population_aggregate

A population also the following methods.
They'll be named here and discussed in place of the method.
1) next_step
    To move the population to the next time step and update accordingly
2) plot_position
    To plot the live position of all people in the population along with their stats.
    Their stats are color coded as usual.
3) plot_stats
    to plot population aggragates live. The stats are color coded as usual
"""
""" ---------------------------------------------"""

class population:
    def __init__(self,city_inst):
        self.total_people = total
        self.population = [person(city_inst,i) for i in range(total)]

    def next_step(self):
        for i in self.population:
# stats

            curr_status = i.status
            curr_infected = i.infected
            curr_recovered = i.recovered
# Position
            if curr_status == False :
                continue
            curr_position = i.position
            next_position = {   "x":  (i.position["x"] + ((0.5 - rand()) * avg_velocity))%city_size["length"],\
                                "y":  (i.position["y"] + ((0.5 - rand()) * avg_velocity))%city_size["width"] }
            i.update_position(next_position)

#Infection
            if curr_status == True and curr_infected == False and curr_recovered == False:

                is_infected = False

                for j in self.population:
                    if j.infected == True:
                        chance_of_infection = prob_infection * (e ** (-1 * distance(i,j)/rad_infection))
                        is_infected = True if rand() <= chance_of_infection else False
                        break
                if is_infected == True:
                    i.time_since_infection +=1
                    i.update_status(True)
                    i.update_infected(True)
                    i.update_recovered(False)
#Status/Recovered
            if curr_status ==True and curr_infected == True and curr_recovered == False:
                is_dead = False
                is_recovered = False

                chance_of_death  =  normalising_factor * ((1 / (std_sickness * (sqrt(2*pi))))\
                                    * ( e ** (-0.5 * (((i.time_since_infection / time_steps_in_day ) - sickest_time )/std_sickness)**2)))/time_steps_in_day

                is_dead = True if rand() <= chance_of_death else False

                if (i.time_since_infection / time_steps_in_day) >= max_time_to_recovery_days:
                    is_recovered = True

                if is_recovered == True:
                    i.update_infected(False)
                    i.update_status(True)
                    i.update_recovered(True)

                if is_dead == True :
                    i.update_infected(False)
                    i.update_status(False)
                    i.update_recovered(False)

                if is_dead == False and is_recovered == False :
                    i.update_infected(True)
                    i.update_status(True)
                    i.update_recovered(False)
                i.time_since_infection += 1
        return self.population




    def plot_position(self):
        positions =[]
        for i in self.population:
            positions.append({  "position":(i.position),\
                                "number":i.number,\
                                "infected":i.infected,\
                                "status":i.status,\
                                "recovered":i.recovered})

        #remove this
        for i in positions:
            print(i["position"])
        #add plot
        X = [i["position"]["x"] for i in positions]
        Y = [i["position"]["y"] for i in positions]
        I = [i["infected"] for i in positions]
        S = [i["status"] for i in positions]
        R = [i["recovered"] for i in positions]

        colors = []
        for i in positions:
            if i["status"] == False:
                colors.append("k")
            elif i["infected"] == True and i["recovered"] == False:
                colors.append("r")
            elif i["recovered"] == True :
                colors.append("b")
            else:
                colors.append("g")

        plt.ion()
        city_map = plt.figure(1)

        plt.clf()
        plt.xlim(0,city_size["length"])
        plt.ylim(0,city_size["width"])

        plt.scatter(X,Y,color = colors)
        city_map.canvas.draw_idle()
        plt.pause(time_step)

    def plot_stats(self):
        positions = []
        for i in self.population:
            positions.append({  "position":(i.position),\
                                "number":i.number,\
                                "infected":i.infected,\
                                "status":i.status,\
                                "recovered":i.recovered})


        #add plot
        S = sum([i["status"] for i in positions])
        I = sum([i["infected"] for i in positions])
        R = sum([i["recovered"] for i in positions])

        susceptible_num = total - (total - S) - I - R
        infected_num = I
        recovered_num = R
        deceased_num = total - S

        labels = ["susceptible","infected","recovered","deceased"]
        values = [susceptible_num,infected_num,recovered_num,deceased_num]

        plt.ion()
        graph = plt.figure(2)


        plt.clf()
        plt.ylim(0,total)
        plt.bar(labels,values,color = ['g','r','b','k'],width = 0.4)
        graph.canvas.draw_idle()
        plt.pause(time_step)

        pop_stats = {   "susceptible":susceptible_num,\
                        "infected":infected_num,\
                        "recovered":recovered_num,\
                        "deceased":deceased_num}
        return pop_stats

""" ---------------------------------------------"""

"""
MAIN STARTS HERE
"""
""" ---------------------------------------------"""

simulate()
plot_aggregate()
sample_inst = sample(day_of_test,num_samples)
results = test_sample(sample_inst)
test_results()
