import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time
from copy import deepcopy


class Node(object):
    def __init__(self, state, parent, depth, path_cost):
        self.state = state
        self.parent = parent
        self.depth = 0
        self.path_cost = path_cost

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.path_cost < node.path_cost


class EightPuzzleSolver():
    def __init__(self, init_state, fin_state, n):
        self.n = n
        self.init_state = init_state
        self.fin_state = fin_state
        self.dx = [1, -1, 0, 0]
        self.dy = [0, 0, 1, -1]
        self.path = []
        self.par = dict()
        self.vis = []

    def same_state(self, state_a, state_b):
        for i in range(self.n):
            for j in range(self.n):
                if state_a[i][j] != state_b[i][j]:
                    return False
        return True

    def next_states(self, state):
        next_states = []
        for i in range(self.n):
            for j in range(self.n):
                if state[i][j] == 0:
                    for k in range(4):
                        if i + self.dx[k] >= 0 and i + self.dx[k] < self.n and j + self.dy[k] >= 0 and j + self.dy[k] < self.n:
                            new_grid = deepcopy(state)
                            new_grid[i][j] = new_grid[i +
                                                      self.dx[k]][j + self.dy[k]]
                            new_grid[i + self.dx[k]][j + self.dy[k]] = 0
                            next_states.append(new_grid)
        return next_states

    def is_goal(self, state):
        return self.same_state(state, self.fin_state)

    def misplaced_tiles(self, state):
        cnt = 0
        for i in range(self.n):
            for j in range(self.n):
                if state[i][j] != self.fin_state[i][j] and state[i][j] != 0:
                    cnt += 1
        return cnt

    def manhattan_dist(self, state):
        dists = []
        res = 0
        for i in range(self.n):
            for j in range(self.n):
                dists.append(self.fin_state[i][j])
        for i in range(self.n):
            for j in range(self.n):
                cur_val = state[i][j]
                x_coor = i
                y_coor = j
                idx = dists.index(cur_val)
                x_gl, y_gl = idx // self.n, idx % self.n
                if cur_val != 0:
                    res += abs(x_coor - x_gl) + abs(y_coor - y_gl)
        return res

    def f_distance(self, state):
        return self.misplaced_tiles(state) + self.manhattan_dist(state)

    def f_misplaced_tiles(self, Node):
        g_x = Node.depth
        h_x = self.misplaced_tiles(Node.state)
        return g_x + h_x

    def f_manhattan_state(self, Node):
        g_x = Node.depth
        h_x = self.manhattan_dist(Node.state)
        return g_x + h_x

    def path_to_goal(self):
        cur_state = self.fin_state
        while not self.same_state(cur_state, self.init_state):
            self.path.append(cur_state)
            hashed = self.hashing(cur_state)
            cur_state = self.par[hashed]
        self.path.append(self.init_state)
        self.path.reverse()
        for i in self.path:
            print(i)
        return self.path

    def hashing(self, state):
        return tuple(tuple(y) for y in set(tuple(x) for x in state))

    def check_explored(self, state):
        for i in self.vis:
            if self.same_state(i, state):
                return True
        return False

    def A_star(self, heuristic='misplaced_tiles'):
        fringe = []
        cnt = 0
        init_node = Node(self.init_state, None, 0, 0)
        self.vis.append(init_node.state)
        if heuristic == 'misplaced_tiles':
            init_node.path_cost = self.f_misplaced_tiles(init_node)
        elif heuristic == 'manhattan_dist':
            init_node.path_cost = self.f_manhattan_state(init_node)
        fringe.append(init_node)
        while True:
            cur_node = fringe.pop()
            if cnt == 0:
                cnt = 1
            if self.is_goal(cur_node.state):
                print("Success")
                print("Cost: ", cur_node.path_cost)
                self.path_to_goal()
                return True
            next_possible_states = self.next_states(cur_node.state)
            for i in next_possible_states:
                if self.check_explored(i) == False:
                    child_path_cost = 0
                    child_node = Node(i, cur_node, cur_node.depth + 1, 0)
                    if heuristic == 'misplaced_tiles':
                        child_path_cost = self.f_misplaced_tiles(child_node)
                    elif heuristic == 'manhattan_dist':
                        child_path_cost = self.f_manhattan_state(child_node)
                    child_node.path_cost = child_path_cost
                    print("Child path cost: ", child_path_cost)
                    fringe.append(child_node)
                    child_hashed = self.hashing(child_node.state)
                    self.vis.append(child_node.state)
                    self.par[child_hashed] = cur_node.state
        return False


init_state = np.array([[1, 3, 2], [4, 6, 8], [0, 7, 5]])
fin_state = np.array([[1, 3, 2], [4, 6, 8], [7, 5, 0]])


NewPuzzleSolver = EightPuzzleSolver(init_state, fin_state, 3)
tic = time.time()
NewPuzzleSolver.A_star(heuristic = 'manhattan_dist')
toc = time.time()
print("Time taken: ", toc - tic, " seconds")
