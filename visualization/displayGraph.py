import numpy as np
import matplotlib.pyplot as plt

import copy
import sys
import os

def readGraph():

    V = []
    E = []

    with open(filepath) as fp:
        # vertex
        line = fp.readline()
        cntV = 0
        cntE = 0
        while line:

            # vertex
            line = line.strip('\n')
            data = line.split(",")
            # print (data)
            V.append([float(data[0]), float(data[1])])
            # V[cntV][0] = data[0]
            # V[cntV][1] = data[1]

            # edges
            line = fp.readline()
            
            if line:

                line = line.strip('\n')
                data = line.split(" ")

                for item in data:
                    if item:
                        # print(item)
                        thing = item.split(",")
                        # vertex read in
                        # E.append([V[cntV][0], V[cntV][1]])
                        # # # E[cntE][0][0] = V[cntV][0]
                        # # # E[cntE][0][1] = V[cntV][1]
                        # # # edge read in 
                        # E.append([thing[0], thing[1]])
                        # # # E[cntE][1][0] = item[0]
                        # # # E[cntE][1][1] = item[1]

                        E.append([[V[cntV][0], V[cntV][1]], [float(thing[0]), float(thing[1])]])
                        cntE += 1
                # print ('\n')

                line = fp.readline()

            cntV += 1

    return V,E




def printGraph(V,E):

    i = 0
    while i < len(V):
        print( V[i][0], V[i][1])
        i += 1

    j = 0
    while j < len(E):
        print( E[j][0][0], E[j][1][0])
        print( E[j][0][1], E[j][1][1])
        j += 1



if __name__ == '__main__':

    filepath = sys.argv[1]
       
    if not os.path.isfile(filepath):
       print("File path {} does not exist. Exiting...".format(filepath))
       sys.exit()

    startX = 0.1
    startY = 0.1

    goalX = 0.9
    goalY = 0.9
   
    fig, ax = plt.subplots(figsize=(5,5), dpi=200)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # draw start
    start = plt.Circle((startX, startY), 0.02, color='g')
    ax.add_artist(start)

    # draw goal
    goal = plt.Circle((goalX, goalY), 0.02, color='r')
    ax.add_artist(goal)

    # draw obstacle
    obstacle = plt.Circle((0.5, 0.5), 0.15, color='black')
    ax.add_artist(obstacle)

    # read in the graph
    V,E = readGraph()

    # printGraph(V,E)

    # add start and end to graph

    # find closest to start and end

    render = 0
    i = 0
    # draw nodes

    while i < len(V):
    #     print( V[i][0], V[i][1])
        new_pt = plt.Circle((V[i][0], V[i][1]), 0.01, color='b')
        ax.add_artist(new_pt)
        i += 1
    #     title = "./prm_frames/{:03d}.png".format(render)
    #     print(title)
    #     fig.savefig(title)
    #     render += 1

    j = 0
    while j < len(E):
    #     print( E[j][0][0], E[j][1][0])
    #     print( E[j][0][1], E[j][1][1])

        new_line = plt.Line2D([E[j][0][0], E[j][1][0]],
                              [E[j][0][1], E[j][1][1]])
        ax.add_artist(new_line)
    #     # fig.savefig("./prm_frames/{:03d}.png".format(render))
        j += 1
    #     render += 1


    plt.show()

    plt.close(fig)

    
