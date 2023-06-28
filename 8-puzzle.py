

import numpy as np
import matplotlib.pyplot as plt
import math
from copy import deepcopy
import random
from queue import PriorityQueue
from heapq import heappush, heappop
import time


class Node(object):
    def __init__(self, state, parent, depth, path_cost):
        self.state = state
        self.parent = parent
        self.depth = depth
        self.path_cost = path_cost

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.path_cost < node.path_cost


class PuzzleSolver():
    def __init__(self, start_state, end_state):
        self.n = start_state.shape[0]
        self.dx = [-1, 1, 0, 0]
        self.dy = [0, 0, 1, -1]
        self.start_state = start_state
        self.end_state = end_state
        self.path = []
        self.par = dict()
        self.vis = []
        self.stack = []
        self.queue = []
        self.depth = dict()

    def next_state(self, grid):
        next_states = []
        for i in range(self.n):
            for j in range(self.n):
                if grid[i][j] == 0:
                    for k in range(4):
                        if i + self.dx[k] >= 0 and i + self.dx[k] < self.n and j + self.dy[k] >= 0 and j + self.dy[k] < self.n:
                            new_grid = deepcopy(grid)
                            new_grid[i][j] = new_grid[i +
                                                      self.dx[k]][j + self.dy[k]]
                            new_grid[i + self.dx[k]][j + self.dy[k]] = 0
                            next_states.append(new_grid)
                    return next_states

    def is_goal(self, cur_state):
        return self.same_state(cur_state, self.end_state)

    def clear_data(self):
        self.path.clear()
        self.stack.clear()
        self.queue.clear()
        self.vis.clear()
        self.depth.clear()
        self.par.clear()

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

    def hashing(self, state):
        # hash the array into a tuple
        # to use it as a key in the dicts
        return tuple(tuple(y) for y in set(tuple(x) for x in state))

    def same_state(self, state_a, state_b):
        for i in range(self.n):
            for j in range(self.n):
                if (state_a[i][j] != state_b[i][j]):
                    return False
        return True

    def check_vis(self, cur_state):
        for i in self.vis:
            if self.same_state(cur_state, i):
                return True
        return False

    def dfs(self):
        self.stack.append(self.start_state)
        self.vis.append(self.start_state)
        while (self.stack):
            cur_state = self.stack.pop()
            if (self.is_goal(cur_state)):
                print("Solution found!")
                return True
            moves = self.next_state(cur_state)
            for i in moves:
                if self.check_vis(i) == False:
                    self.stack.append(i)
                    self.vis.append(i)
                    hash_state = self.hashing(i)
                    self.par[hash_state] = cur_state
        return False

    def bfs(self):
        self.clear_data()
        self.queue.append(self.start_state)
        self.vis.append(self.start_state)
        while self.queue:
            cur_state = self.queue.pop(0)
            if self.is_goal(cur_state):
                print("Solution found!")
                return True
            moves = self.next_state(cur_state)
            for i in moves:
                if self.check_vis(i) == False:
                    self.queue.append(i)
                    self.vis.append(i)
                    hash_state = self.hashing(i)
                    self.par[hash_state] = cur_state

        return False

    def limited_depth_dfs(self, max_depth=3):
        self.clear_data()
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
                    if self.check_vis(i) == False:
                        self.stack.append(i)
                        self.vis.append(i)
                        hash_state = self.hashing(i)
                        self.par[hash_state] = cur_state
                        self.depth[hash_state] = self.depth[cur_hash] + 1
        return False

    def iterative_deepening_dfs(self, max_depth=100):
        for i in range(max_depth):
            self.clear_data()
            if self.limited_depth_dfs(max_depth=i) == True:
                return True
        return False

    def naive_cost(self, state):
        # how many misplaced tiles ?
        cnt = 0
        for i in range(self.n):
            for j in range(self.n):
                if (state[i][j] != self.end_state[i][j]) and state[i][j] != 0:
                    cnt += 1
        return cnt

    def manhattan_cost(self, state):
        dist = 0
        for i in range(self.n):
            for j in range(self.n):
                if state[i][j] != 0:
                    x, y = divmod(state[i][j] - 1, self.n)
                    dist += abs(x - i) + abs(y - j)
        return dist

    def branch_and_bound(self):
        self.clear_data()
        pq = PriorityQueue()
        pqz = Node(self.start_state, 0, 0,
                   self.manhattan_cost(self.start_state))
        pq.put(pqz)
        ini_cost = self.manhattan_cost(self.start_state)
        z = self.hashing(self.start_state)
        self.vis.append(self.start_state)
        while not pq.empty():
            cur_node = pq.get()
            if self.is_goal(cur_node.state):
                print("Solution found!")
                print("Distance covered: ", cur_node.depth)
                return True
            moves = self.next_state(cur_node.state)
            for i in moves:
                if self.check_vis(i) == False:
                    hash_state = self.hashing(i)
                    self.par[hash_state] = cur_node.state
                    pq.put(
                        Node(i, self.par[hash_state], cur_node.depth + 1, self.manhattan_cost(i)))
                    self.vis.append(i)
        return False

    def best_first_search(self):
        self.clear_data()
        pq = PriorityQueue()
        pqz = Node(self.start_state, 0, 0, self.naive_cost(self.start_state))
        pq.put(pqz)
        ini_cost = self.naive_cost(self.start_state)
        z = self.hashing(self.start_state)
        self.vis.append(self.start_state)
        while not pq.empty():
            cur_node = pq.get()
            if self.is_goal(cur_node.state):
                print("Solution found!")
                print("Distance covered: ", cur_node.depth)
                return True
            moves = self.next_state(cur_node.state)
            for i in moves:
                if self.check_vis(i) == False:
                    hash_state = self.hashing(i)
                    self.par[hash_state] = cur_node.state
                    pq.put(
                        Node(i, self.par[hash_state], cur_node.depth + 1, self.naive_cost(i)))
                    self.vis.append(i)
        return False

    def hill_climbing_search(self):
        self.clear_data()
        pq = list()
        pqz = Node(self.start_state, 0, 0, self.naive_cost(self.start_state))
        pq.append(pqz)
        ini_cost = self.naive_cost(self.start_state)
        z = self.hashing(self.start_state)
        self.vis.append(self.start_state)
        while pq:
            cur_node = pq.pop()
            cur_goal = self.naive_cost(cur_node.state)
            if self.is_goal(cur_node.state):
                print("Solution found!")
                print("Distance covered: ", cur_node.depth)
                return True
            moves = self.next_state(cur_node.state)
            temp_pq = PriorityQueue()
            for i in moves:
                if self.check_vis(i) == False:
                    hash_state = self.hashing(i)
                    self.par[hash_state] = cur_node.state
                    temp_pq.put(
                        Node(i, self.par[hash_state], cur_node.depth + 1, self.naive_cost(i)))
                    self.vis.append(i)
            while (not temp_pq.empty()):
                zz = temp_pq.get()
                pq.insert(0, zz)
        return False

    def beam_search(self, k):
        self.clear_data()
        pq = list()
        pqz = Node(self.start_state, 0, 0, self.naive_cost(self.start_state))
        pq.append(pqz)
        ini_cost = self.naive_cost(self.start_state)
        z = self.hashing(self.start_state)
        self.vis.append(self.start_state)
        while pq:
            f = min(k, len(pq))
            Xk = list()
            for i in range(f):
                cur_node = pq.pop(0)
                cur_goal = self.naive_cost(cur_node.state)
                Xk.append(cur_node)
            while pq:
                pq.pop()
            for ii in Xk:
                if self.is_goal(ii.state):
                    print("Solution found!")
                    print("Distance covered: ", ii.depth)
                    return True
                moves = self.next_state(ii.state)
                for jj in moves:
                    if self.check_vis(jj) == False:
                        hash_state = self.hashing(jj)
                        self.par[hash_state] = ii.state
                        pq.append(
                            Node(jj, self.par[hash_state], ii.depth + 1, self.naive_cost(jj)))
                        self.vis.append(jj)
                pq.sort(key=lambda x: x.path_cost)
        return False

    def solve(self, alg):
        if alg == "DFS":
            if self.dfs() == True:
                self.path_to_goal()
            else:
                print("No solution found!")
        elif alg == "BFS":
            if self.bfs() == True:
                self.path_to_goal()
            else:
                print("No solution found!")
        elif alg == "LDFS":
            if self.limited_depth_dfs() == True:
                self.path_to_goal()
            else:
                print("No solution found!")
        elif alg == "IDDFS":
            if self.iterative_deepening_dfs() == True:
                self.path_to_goal()
            else:
                print("No solution found!")
        elif alg == "BNB":
            if self.branch_and_bound() == True:
                self.path_to_goal()
            else:
                print("No solution found!")
        elif alg == "BEST":
            if self.best_first_search() == True:
                self.path_to_goal()
            else:
                print("No solution found!")
        elif alg == "HILL":
            if self.hill_climbing_search() == True:
                self.path_to_goal()
            else:
                print("No solution found!")
        elif alg == "BEAM":
            if self.beam_search(2) == True:
                self.path_to_goal()
            else:
                print("No solution found!")


start_state = np.array([[1, 3, 2], [4, 6, 8], [0, 7, 5]])
end_state = np.array([[1, 3, 2], [4, 6, 8], [7, 5, 0]])


NewPuzzleSolver = PuzzleSolver(start_state, end_state)
tic = time.time()
NewPuzzleSolver.solve("BEST")
toc = time.time()
print("Time taken: ", toc - tic, " seconds")
