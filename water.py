import numpy as np
import math 
import random
from copy import deepcopy
from queue import PriorityQueue
from heapq import heappush, heappop
import time


class WaterBottleSolver():
    def __init__(self, start_state, end_state, max_a, max_b):
        self.start_state = start_state
        self.end_state = end_state
        self.max_a = max_a
        self.max_b = max_b
        self.path = []
        self.par = dict()
        self.vis = []
        self.stack = []
        self.queue = []
        self.depth = dict()
        
    def next_state(self, state):
        next_states = []
        cur_a, cur_b = state
        next_states.append((0, cur_b))
        next_states.append((cur_a, 0))
        next_states.append((self.max_a, cur_b))
        next_states.append((cur_a, self.max_b))
        if cur_a + cur_b <= self.max_b:
            next_states.append((cur_a + cur_b, self.max_b))
        if cur_a + cur_b <= self.max_a:
            next_states.append((self.max_a, cur_a + cur_b))
        if cur_a - cur_b >= 0:
            next_states.append((cur_a - cur_b, 0))
        if cur_b - cur_a >= 0:
            next_states.append((0, cur_b - cur_a))
        return next_states
    
    def is_goal(self, state):
        return (state[0] == self.end_state[0] and state[1] == self.end_state[1])
    
    def check_visited(self, state):
        for i in self.vis:
            if i[0] == state[0] and i[1] == state[1]:
                return True
        return False
    
    def hashing(self, state):
        return tuple(state)
    
    def path_to_goal(self):
        cur_state = self.end_state
        while not self.same_state(cur_state, self.start_state):
            self.path.append(cur_state)
            hashed = self.hashing(cur_state)
            cur_state = self.par[hashed]
        self.path.append(self.start_state)
        self.path.reverse()
        for i in self.path:
            print(i)
        return self.path

    def same_state(self, state1, state2):
        return (state1[0] == state2[0] and state1[1] == state2[1])
    
    def dfs_solver(self):
        self.stack.append(self.start_state)
        self.vis.append(self.start_state)
        while (self.stack):
            cur_state = self.stack.pop()
            if (self.is_goal(cur_state)):
                print("Solution found!")
                return True
            moves = self.next_state(cur_state)
            for i in moves:
                if self.check_visited(i) == False:
                    self.stack.append(i)
                    self.vis.append(i)
                    hashed_state = self.hashing(i)
                    self.par[hashed_state] = cur_state
            return False
    
    def bfs_solver(self):
        self.queue.append(self.start_state)
        self.vis.append(self.start_state)
        while self.queue:
            cur_state = self.queue.pop(0)
            if self.is_goal(cur_state):
                print("Solution found!")
                return True
            moves = self.next_state(cur_state)
            for i in moves:
                if self.check_visited(i) == False:
                    self.queue.append(i)
                    self.vis.append(i)
                    hash_state = self.hashing(i)
                    self.par[hash_state] = cur_state
        return False
    
    def depth_first_search(self, state, depth):
        self.depth[state] = depth
        if self.is_goal(state):
            print("Solution found!")
            return True
        moves = self.next_state(state)
        for i in moves:
            if self.check_visited(i) == False:
                self.vis.append(i)
                hashed_state = self.hashing(i)
                self.par[hashed_state] = state
                if self.depth_first_search(i, depth + 1):
                    return True
        return False
    
    def limited_depth_search(self, max_depth = 3):
        self.stack.append(self.start_state)
        self.vis.append(self.start_state)
        z = self.hashing(self.start_state)
        self.depth[z] = 0
        while self.stack:
            cur_state = self.stack.pop()
            cur_hash = self.hashing(cur_state)
            if self.is_goal(cur_state):
                print("Solution found!")
                return True
            if (self.depth[cur_hash] <= max_depth):
                for i in self.next_state(cur_state):
                    if self.check_visited(i) == False:
                        self.stack.append(i)
                        self.vis.append(i)
                        hashed_state = self.hashing(i)
                        self.par[hashed_state] = cur_state
                        self.depth[hashed_state] = self.depth[cur_hash] + 1
        return False   
        
    
    def solver_menu(self, alg = "DFS"):
        if alg == "DFS":
            self.dfs_solver()
        elif alg == "BFS":
            self.bfs_solver()
        elif alg == "LDS":
            self.limited_depth_search()
        elif alg == "IDS":
            for i in range(1, 100):
                print("Depth: ", i)
                if self.depth_first_search(self.start_state, 0):
                    return True
        else:
            print("Invalid algorithm")
            return False
        self.path_to_goal()
        return True

water = WaterBottleSolver((0, 0), (3, 2), 4, 3)
water.solver_menu("BFS")