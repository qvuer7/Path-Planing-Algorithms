import random

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
from heapq import nsmallest
from collections import defaultdict
import time

class Graph():
    def __init__(self):
        self.edges = defaultdict(list)
        self.weights = {}


    def add_edge(self, from_node, to_node, weight):
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.weights[(from_node, to_node)] = weight
        self.weights[(to_node, from_node)] = weight

class node:
    """
    Node class
    """

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.parent = None
        self.nearest = []
        self.distance = None
def get_random_node(minRand, maxRand):
    """
        Generate totally random value
    """
    xx = random.uniform(minRand, maxRand)
    yy = random.uniform(minRand, maxRand)
    zz = random.uniform(minRand, maxRand)

    rnode = node(xx, yy, zz)
    return rnode
def cal_dist_and_angle(nodef, nodet):
    """
    Cartesian calculations generates newnode according to step
    """
    newnode = node(0, 0, 0)
    dx = nodet.x - nodef.x
    dy = nodet.y - nodef.y
    dz = nodet.z - nodef.z
    r = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    if r != 0:
        xx = dx * (step / r)
        yy = dy * (step / r)
        zz = dz * (step / r)
        newnode = node(nodef.x + xx, nodef.y + yy, nodef.z + zz)

    return newnode, r
def check_between(nodef, nodet):
    while True:
        newnodee = cal_dist_and_angle(nodef, nodet)[0]
        if check_inner_outer(newnodee, obstaclelist):

            return False
        else:
            nodef = newnodee
        if cal_dist_and_angle(newnodee, nodet)[1] <= step:

            return True  # $ # returns True if points are avaliable to connection
def check_inner_outer(node, list):
    for i in range(len(list)):
        dx = node.x - list[i][0]
        dy = node.y - list[i][1]
        dz = node.z - list[i][2]
        r = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        if r <= list[i][3]:
            return True  # return false node is not in obstacle, return true if node is in obstacle
    return False  # use TRUE to check if, and FALSE to check if not

def sct_obst(obstaclenodelist, color):
    """
     Plotting obstalces
    """
    for i in range(len(obstaclenodelist)):
        ax.scatter(obstaclenodelist[i].x, obstaclenodelist[i].y, obstaclenodelist[i].z, color=color)
def obstacle_coordinates(list):
    """
    recieving all obstacles coordinates
    """
    a = 10
    b = 10
    u = np.linspace(0, 2 * np.pi, a)
    v = np.linspace(0, np.pi, b)

    xx = [0 for i in range(len(list))]
    yy = [0 for i in range(len(list))]
    zz = [0 for i in range(len(list))]
    obstaclenodes = []
    for i in range(len(list)):
        xx[i] = list[i][3] * np.outer(np.cos(u), np.sin(v)) + list[i][0]
        yy[i] = list[i][3] * np.outer(np.sin(u), np.sin(v)) + list[i][1]
        zz[i] = list[i][3] * np.outer(np.ones(np.size(u)), np.cos(v)) + list[i][2]
        for ii in range(a):
            for iii in range(b):
                obstaclenodes.append(node(xx[i][ii][iii], yy[i][ii][iii], zz[i][ii][iii]))

    return obstaclenodes
def get_random_list(iterations, obstaclelistt):
    list = []
    for i in range(iterations):
        randomnode = get_random_node(minRand, maxRand)
        if check_inner_outer(randomnode, obstaclelistt) == False:
            list.append(randomnode)

        else:
            i = i - 1
    return list
def dijsktra(graph, initial, end):

    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()

    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            weight = graph.weights[(current_node, next_node)] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)

        next_destinations = {nodee: shortest_paths[nodee] for nodee in shortest_paths if nodee not in visited}
        if not next_destinations:
            return "Route Not Possible"
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])

    # Work back through destinations in shortest path
    path = []
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        current_node = next_node
    # Reverse path
    path = path[::-1]
    return path
def k_nearest_nodes(list,node):
    for i in range(len(list)):
        r = []
        nearestNodesNumber = []
        nearestNodes = []
        NN = []
        for b in range(len(list)):
            r.append(cal_dist_and_angle(node, list[b])[1])  # all node distances

            kMin = nsmallest(k + 1, r)
            kMin.remove(kMin[0])
        for l in range(len(kMin)):
            nearestNodesNumber.append(r.index(kMin[l]))
            nearestNodes.append(list[nearestNodesNumber[l]] ) # k nearest nodes
            avaliable = check_between(list[i], nearestNodes[l])
            if avaliable:
                NN.append(nearestNodes[l])
    return NN,kMin



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
step = 0.9
minRand = -5
maxRand = 25
graph = Graph()
obstaclelist = [[10,10,10,5]]
obstaclenodes = obstacle_coordinates(obstaclelist)
iterations = 100




randomnodelist = get_random_list(iterations,obstaclelist)

snode = node(0,0,0)
gnode = node(20,20,20)
randomnodelist.append(gnode)
randomnodelist.insert(0,snode)
k = len(randomnodelist)









edges = []
c = randomnodelist.copy()
l = len(c)
p = 0
while l>0:

    NN = k_nearest_nodes(c,c[0])[0]
    distances = k_nearest_nodes(c,c[0])[1]

    for i in range(len(NN)):
        edges.append((c[0], NN[i],distances[i]))
        print((c[0], NN[i],distances[i]))
    c.remove(c[0])
    l = len(c)

NNlast = k_nearest_nodes(randomnodelist, randomnodelist[len(randomnodelist) - 1])[0]
distancesLast = k_nearest_nodes(randomnodelist, randomnodelist[len(randomnodelist) - 1])[1]

for i in range(len(NNlast)):
    edges.append((randomnodelist[len(randomnodelist) - 1], NNlast[i], distancesLast[i]))

for edge in edges:
    graph.add_edge(*edge)


NNo = k_nearest_nodes(randomnodelist,randomnodelist[2])[0]




pathh = dijsktra(graph,snode,gnode)
sct_obst(randomnodelist,color = 'g')
sct_obst(pathh, color = 'r')
sct_obst(obstaclenodes, color = 'b')
plt.show()



















