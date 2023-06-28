import numpy as np
import random

n = int(input())

dx = [-1, 1, 0, 0]
dy = [0, 0, 1, -1]


values = range(1, n * n + 1)
rand_matrix = random.sample(values, n * n)
rand_matrix = np.array(rand_matrix).reshape(n, n)

def switch(i, j):
    if i < 0 or i >= n or j < 0 or j >= n:
        print("Invalid move")
        return
    for k in range(4):
        if rand_matrix[i + dx[k]][j + dy[k]] == 0:
            rand_matrix[i + dx[k]][j + dy[k]] = rand_matrix[i][j]
            rand_matrix[i][j] = 0
            return



for i in range(n):
    for j in range(n):
        if rand_matrix[i][j] == n * n:
            rand_matrix[i][j] = 0

while (True):
    print(rand_matrix)
    print("Do you want to quit? (y/n)")
    if input() == 'y':
        break
    print("Enter the number you want to move: ")
    num = int(input())
    for i in range(n):
        for j in range(n):
            if rand_matrix[i][j] == num:
                switch(i, j)
                break
        else:
            continue
        break
    else:
        print("Invalid move")
        continue
