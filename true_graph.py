import numpy as np
import math
import random
from copy import deepcopy
from queue import PriorityQueue
from heapq import heappush, heappop
import time

class Graph():
    def __init__ (self, n, adj_list):
        self.v = n
        self.edges = [[-1 for i in range(n)] for j in range(n)]
        self.visited = []
    
    def add_edge(self, u, v, w):
        self.edges[u][v] = w
        self.edges[v][u] = w
        

class GraphSolver():
    def __init__(self, graph, start, end):
        self.graph = Graph(graph.v, graph.edges)
        self.start = start
        self.end = end
        self.path = []
        self.stack = []
        self.queue = []
        self.pq = PriorityQueue()
        self.vis = dict()
        self.par = dict()
    
    def path_to_goal(self):
        cur_state = self.end
        while cur_state != self.start:
            self.path.append(cur_state)
            cur_state = self.par[cur_state]
        self.path.append(self.start)
        self.path.reverse()
        for i in self.path:
            print(i, end = " ")
       
    def BestFirstSearchSolver(self):
        self.pq.append(self.start)
        self.vis.append(self.start)
        vis_mark = [False] * self.graph.v
        self.pq.put((0, self.start))
        vis_mark[self.start] = True
        
        while self.pq.empty() == False:
            u = self.pq.get()[1]
            if u == self.end:
                break
            for v, c in self.graph.edges[u]:
                if vis_mark[v] == False:
                    vis_mark[v] = True
                    self.pq.put((c, v))
                    self.par[v] = u
    
    def GreedyFirstSearchSolver(self):
         
            
        
        
        