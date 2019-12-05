import math
import random

import matplotlib.pyplot as plt
import numpy as np

from RRT import Smoothness
from mpl_toolkits.mplot3d import Axes3D

class node:
    """
    Node class
    """

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.parent = None


def inputt():
    try:
        iterations = int(input("enter number of iterations, (by default = 1000) =>"))
    except ValueError:
        iterations = 1000
    try:
        numberofObjects = int(input("enter number of objects, (by default 1) =>"))
    except ValueError:
        numberofObjects = 1
    try:
        DroneSize = int(input("enter drone safe radius, (by default = 0.5) =>"))
    except ValueError:
        DroneSize = 0.5

    try:
        step = int(input("enter drone step, (by default = 1) =>"))
    except ValueError:
        step = 1

    try:
        numberOfObstacles = int(input("Enter number of obstacles, (by default 3) =>"))
    except ValueError:
        numberOfObstacles = 3


    goal = [[None for j in range(3)] for i in range(numberofObjects)]
    start = [[None for j in range(3)] for i in range(numberofObjects)]
    obstaclelist = [[0 for j in range(4)] for i in range(numberOfObstacles)]

    for i in range(0, numberofObjects):
        for g in range(0, 3):
            start[i][g] = int(input(f"input {i + 1} start coordinates =>"))

    for i in range(0, numberofObjects):
        for g in range(0, 3):
            goal[i][g] = int(input(f"input {i + 1} goal coordinates =>"))

    for i in range(0, numberOfObstacles):
        for g in range(0, 4):
            if g != 3:
                obstaclelist[i][g] = int(input(f"input {i + 1} obstacle coordinate =>"))
            else:
                obstaclelist[i][g] = int(input(f"input {i + 1} obstacle radius =>"))
    obstaclenodes = obstacle_coordinates(obstaclelist)
    return iterations, numberofObjects, DroneSize, step, numberOfObstacles, start, goal, obstaclenodes


def obstacle_coordinates(obstaclelist):
    """
    recieving all obstacles coordinates
    """
    a = 20
    b = 20
    u = np.linspace(0, 2 * np.pi, a)
    v = np.linspace(0, np.pi, b)

    xx = [0 for i in range(len(obstaclelist))]
    yy = [0 for i in range(len(obstaclelist))]
    zz = [0 for i in range(len(obstaclelist))]
    obstaclenodes = []
    for i in range(len(obstaclelist)):
        xx[i] = obstaclelist[i][3] * np.outer(np.cos(u), np.sin(v)) + obstaclelist[i][0]
        yy[i] = obstaclelist[i][3] * np.outer(np.sin(u), np.sin(v)) + obstaclelist[i][1]
        zz[i] = obstaclelist[i][3] * np.outer(np.ones(np.size(u)), np.cos(v)) + obstaclelist[i][2]
        for ii in range(a):
            for iii in range(b):
                obstaclenodes.append(node(xx[i][ii][iii], yy[i][ii][iii], zz[i][ii][iii]))

    return obstaclenodes


def check_obstacle(node, obstaclenodelist):
    """
    obstacle check
    """
    a = False
    for i in range(len(obstaclenodelist)):
        a = check_collision(node, obstaclenodelist[i])
        if a == True:
            return a
    return a


def check_collision(node1, node2):
    r = cal_dist_and_angle(node1, node2)[1]
    if r <= safesize:
        a = True
    else:
        a = False
    return a


def cal_dist_and_angle(nodef, nodet):
    """
    Cartesian calculations generates newnode according to step
    """

    dx = nodet.x - nodef.x
    dy = nodet.y - nodef.y
    dz = nodet.z - nodef.z

    r = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    xx = dx * (input[3] / r)
    yy = dy * (input[3] / r)
    zz = dz * (input[3] / r)
    newnode = node(nodef.x + xx, nodef.y + yy, nodef.z + zz)

    return newnode, r


def get_random_node(minRand, maxRand, goal_sample_rate, goal):
    """
        Generate totally random value
    """

    if random.randint(0, 100) > goal_sample_rate:
        xx = random.uniform(minRand, maxRand)
        yy = random.uniform(minRand, maxRand)
        zz = random.uniform(minRand, maxRand)
    else:
        xx = goal[0]
        yy = goal[1]
        zz = goal[2]
    rnode = node(xx, yy, zz)
    return rnode


def check_nearest_node(nodelist, node):
    """
      Outputs index of nearest node
    """
    distance = []
    for i in range(len(nodelist)):
        distance.append(cal_dist_and_angle(node, nodelist[i])[1])
    return distance.index(min(distance))


def generate_final_course(gnode, nodelist):
    """
           Generates final course
    """
    path = [[gnode.x, gnode.y, gnode.z]]
    node = nodelist[len(nodelist) - 2]
    while node.parent is not None:
        path.append([node.x, node.y, node.z])
        node = node.parent
    path.append([node.x, node.y, node.z])

    return path


def sct_obst(obstaclenodelist):
    """
     Plotting obstalces
    """
    for i in range(len(obstaclenodelist)):
        ax.scatter(obstaclenodelist[i].x, obstaclenodelist[i].y, obstaclenodelist[i].z, color='b')


input = inputt()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

obstaclenodes = input[7]
sct_obst(obstaclenodes)
safesize = input[2] + input[3]
snode = [node(input[5][i][0], input[5][i][1], input[5][i][2]) for i in range(input[1])]
gnode = [node(input[6][i][0], input[6][i][1], input[6][i][2]) for i in range(input[1])]
listener = [True for i in range(input[1])]
nodelist = [[0 for j in range(1)] for i in range(input[1])]

goal_sample_rate = [2 for i in range(input[1])]
for i in range(0, input[1]):
    ax.scatter(input[6][i][0], input[6][i][1], input[6][i][2], color="r")
    ax.scatter(input[5][i][0], input[5][i][1], input[5][i][2], color="y")
    nodelist[i][0] = snode[i]
minRand = -200
maxRand = 200
minRandd = minRand
maxRandd = maxRand
nodeNear = [0 for i in range(input[1])]
pathSmooth = [0 for i in range(input[1])]
path = [0 for i in range(input[1])]
rnode = [0 for i in range(input[1])]
collision = False
colors = ['r','g','y','k','b', 'm']
for i in range(input[0]):
    for obj in range(input[1]):
        if listener[obj] == False:
            continue
        else:
            rnode[obj] = (get_random_node(minRand, maxRand, goal_sample_rate[obj], input[6][obj]))
            nodeNear[obj] = (check_nearest_node(nodelist[obj], rnode[obj]))
            snode[obj] = cal_dist_and_angle(nodelist[obj][len(nodelist[obj]) - 1], rnode[obj])[0]
            snode[obj].parent = nodelist[obj][nodeNear[obj]]

            for k in range(len(nodelist)):

                if k == obj:
                    collision = check_obstacle(snode[obj], obstaclenodes)
                else:
                    collision = check_obstacle(snode[obj], obstaclenodes)
                    if collision == False:
                        collision = check_collision(snode[k], nodelist[obj][len(nodelist[obj]) - 1])
                    else:
                        continue

            if collision:

                ax.scatter(-2, -2, -2)
                minRand = snode[obj].x - 2
                maxRand = snode[obj].x + 2
                goal_sample_rate[obj] = 2

            else:

                nodelist[obj].append(snode[obj])
                ax.scatter(snode[obj].x, snode[obj].y, snode[obj].z, color=colors[obj])
                minRand = minRandd
                maxRand = maxRandd
                goal_sample_rate[obj] = 100

            if (cal_dist_and_angle(snode[obj], gnode[obj])[1] <= input[3]):
                nodelist[obj].append(gnode[obj])
                listener[obj] = False
                path[obj] = generate_final_course(gnode[obj], nodelist[obj])
                pathSmooth[obj] = Smoothness.smooth(path[obj])
            plt.pause(0.01)

    if not any(listener):
        break

ax.clear()
sct_obst(obstaclenodes)
for i in range(0, len(pathSmooth)):
    for g in range(len(pathSmooth[i])):
        ax.scatter(pathSmooth[i][g][0], pathSmooth[i][g][1], pathSmooth[i][g][2], color='r')

plt.show()

# def cal_length(pathh):
#     r = 0
#     for i in range(len(pathh[0])):
#         if i == len(pathh[0]) - 1 : return r
#         tx = pathh[0][i + 1][0]
#         ty = pathh[0][i + 1][1]
#         tz = pathh[0][i + 1][2]
#         fx = pathh[0][i + 1][0]
#         fy = pathh[0][i + 1][1]
#         fz = pathh[0][i][2]
#         dx = tx - fx
#         dy = ty - fy
#         dz = tz - fz
#         r = r + math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)



