import numpy as np
import math
import random as rand
import time
import copy as cp
import heapq as hq
import sys
import matplotlib.pyplot as plt
# simulated annealing to solve TSP

# running this program executes the TSP 36-city problem with my favourite schedule
# To run:
# Have randTSP folder in the same directory and also have the 36 city file in there
# Then run, `python q1.py`, i ran it on python 3.5.2

# global variable to hold the costs of travel between cities
# as well as city names ie. 0 corresponds to A
costs = []
city_names = []

def simulated_annealing(init_state, sched):
	curr_state = init_state

	t = 1
	times = []
	costs = []
	while True:
		T = sched(t)
		# using 0.0001 for sched = 1/t
		times.append(t)
		costs.append(curr_state.cost)
		if T <= 0.00001:
			return curr_state, times, costs
		new_state = curr_state.random_successor()
		# we want diff to be > 0 if new_state.cost > curr_state.cost
		diff = curr_state.cost - new_state.cost
		if diff >= 0:
			curr_state = new_state
		else:
			# set with probability e^diff/T
			# rand.random() selects a number between 0 and 1
			if (rand.random() < math.exp(diff/T)):
				curr_state = new_state
		t += 1
	return curr_state, times, costs

# represents our TSP
# [5,1,0,4,2,3,5] is one example of a tour 
class State:
	def __init__(self, tour, cost):
		self.tour = tour
		self.cost = cost

	# operator
	def random_successor(self):
		# just select two numbers between 0 and number of cities - 1
		n_cities = len(self.tour)-2
		ind1 = rand.randint(0, n_cities)
		ind2 = rand.randint(0, n_cities)
		new_tour = cp.copy(self.tour)
		# if the two indices we want to swap happen to be the same
		if ind1 == ind2:
			# then again randomly select
			ind2 = rand.randint(0, n_cities)
			# if same again
			if ind1 == ind2:
				# go to next city in the tour
				ind2 = (ind2 + 1)%(n_cities+1)

		# if one of the cities end up being the first in tour,
		# make sure the last gets changed as well
		if ind1 == 0:
			new_tour[n_cities+1] = self.tour[ind2]
		elif ind2 == 0:
			new_tour[n_cities+1] = self.tour[ind1]

		new_tour[ind1] = self.tour[ind2]
		new_tour[ind2] = self.tour[ind1]
		# ATTENTION: THIS IS JUST FOR NOW, tour_cost() DOES NOT NEED TO BE CALLED, CAN JUST PERFORM SOME
		# ARITHMETIC ON THE OLD COST TO GET NEW COST BUT I'M TOO LAZY RIGHT NOW
		# TODO: FIX LATER
		return State(new_tour, tour_cost(new_tour))

# takes in the number of cities
# and returns a random tour
def tsp_init(n):
	tour = np.zeros(n+1) - 1
	for i in range(n):
		j = rand.randint(0, n-1)
		if tour[j] == -1:
			tour[j] = i
		else:
			while tour[j] != -1:
				j = (j + 1) % n
			tour[j] = i

	tour[n] = tour[0]
	return tour

def tour_cost(tour):
	cost = 0
	for i in range(len(tour)-1):
		cost += costs[int(tour[i])][int(tour[i+1])]
	return cost

# the annealing scheduling function as a function of t (time)
# what we want is for the algorithm to explore a lot early and try lots of tours
# then get stricter in the future
# choice 1: 1/t
# choice 2: 100/t^2
# choice 3: 1000-t
def schedule(t):
	#return 1/t
	return 10000/t**2
	#return 10000-t
# reading input file
# and populate the costs variable
def read_file(filepath):
	global costs
	global city_names
	costs = []
	city_names = []
	file = open(filepath, 'r')
	n_cities = int(file.readline())
	coords = []
	for i in range(n_cities):
		new_city = file.readline().split(' ')
		city_names.append(new_city[0])
		coords.append([(int)(new_city[1]), (int)(new_city[2])])

	for i in range(n_cities):
		dist_to_i = []
		for j in range(n_cities):
			if i == j:
				dist_to_i.append(0.)
			else:
				dist_to_i.append(math.sqrt((coords[i][0]-coords[j][0])**2 + 
					(coords[i][1] - coords[j][1])**2))
		costs.append(dist_to_i)
	costs = np.array(costs)
	file.close()

# running on the 16 folders to see which scheduling function is best
'''
for i in range(1,17):
	filepath = "randTSP/" + str(i) + "/"
	cost_avg = 0.
	for j in range(1,11):
		filename = filepath + "instance_" + str(j) + ".txt"
		read_file(filename)
		# creates a random tour
		start_tour = tsp_init(len(city_names))
		# creates a state out of the new tour
		start_state = State(start_tour, tour_cost(start_tour))
		final_state, dummy1, dummy2 = simulated_annealing(start_state, schedule)
		cost_avg += final_state.cost
	cost_avg = cost_avg/10.
	print("folder: %d, avg: %f" % (i, cost_avg))
'''
read_file("randTSP/problem36")
start_tour = tsp_init(len(city_names))
start_state = State(start_tour, tour_cost(start_tour))
final_state, x_times, y_costs = simulated_annealing(start_state, schedule)
print("problem 36")
print(final_state.tour)
print("cost: %f" % final_state.cost)

plt.plot(x_times, y_costs)
plt.title("Time vs. Tour cost")
plt.xscale("log")
plt.xlabel("Time")
plt.ylabel("Tour Cost")
plt.show()
