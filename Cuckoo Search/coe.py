## import copy
import random
import numpy as np

# ----------------------------------------------------
# TIC TAC TOE UTILITIES
# ----------------------------------------------------
def check_winner(board):
    lines = [
        # rows
        [board[0][0], board[0][1], board[0][2]],
        [board[1][0], board[1][1], board[1][2]],
        [board[2][0], board[2][1], board[2][2]],
        # columns
        [board[0][0], board[1][0], board[2][0]],
        [board[0][1], board[1][1], board[2][1]],
        [board[0][2], board[1][2], board[2][2]],
        # diagonals
        [board[0][0], board[1][1], board[2][2]],
        [board[0][2], board[1][1], board[2][0]]
    ]

    for line in lines:
        if line == ["O","O","O"]: return "O"
        if line == ["X","X","X"]: return "X"
    return None

def get_empty_cells(board):
    return [(i,j) for i in range(3) for j in range(3) if board[i][j] == " "]

def is_full(board):
    return all(board[i][j] != " " for i in range(3) for j in range(3))

# ----------------------------------------------------
# FITNESS FUNCTION FOR A FULL MOVE SEQUENCE
# ----------------------------------------------------
def fitness_of_sequence(sequence, init_board):
    board = copy.deepcopy(init_board)
    turn = "O"  # AI plays O

    for (r,c) in sequence:
        if board[r][c] != " ":
            return -999   # illegal move
        board[r][c] = turn

        winner = check_winner(board)
        if winner == "O": return 100
        if winner == "X": return -100

        turn = "X" if turn == "O" else "O"

    # Evaluate final state heuristically
    if check_winner(board) == "O": return 100
    if check_winner(board) == "X": return -100
    return 0

# ----------------------------------------------------
# LEVY FLIGHT PERTURBATION
# ----------------------------------------------------
def levy_flight():
    u = np.random.normal(0,1)
    v = np.random.normal(0,1)
    return u / abs(v)**(1/1.5)

# ----------------------------------------------------
# CUCKOO SEARCH TO FIND BEST MOVE SEQUENCE
# ----------------------------------------------------
def cuckoo_search_best_sequence(board, nests=20, pa=0.25, max_steps=5):
    empty = get_empty_cells(board)

    # generate random sequences
    population = []
    for _ in range(nests):
        steps = random.randint(1, max_steps)
        seq = random.sample(empty, k=min(steps, len(empty)))
        population.append(seq)

    # evaluate fitness
    fitness = [fitness_of_sequence(seq, board) for seq in population]

    for _ in range(30):  # iterations
        best_idx = np.argmax(fitness)
        best_seq = population[best_idx]

        # generate new cuckoo
        new_seq = copy.deepcopy(best_seq)
        if empty:
            if random.random() < 0.7:
                new_seq.append(random.choice(empty))

        # random mutation
        if random.random() < 0.5 and len(new_seq) > 1:
            idx = int(abs(levy_flight())) % len(new_seq)
            new_seq.pop(idx)

        new_fit = fitness_of_sequence(new_seq, board)

        # replace worst nest
        worst_idx = np.argmin(fitness)
        if new_fit > fitness[worst_idx]:
            population[worst_idx] = new_seq
            fitness[worst_idx] = new_fit

        # abandon nests
        for i in range(nests):
            if random.random() < pa:
                s = random.sample(empty, k=min(random.randint(1, max_steps), len(empty)))
                population[i] = s
                fitness[i] = fitness_of_sequence(s, board)

    best_idx = np.argmax(fitness)
    return population[best_idx], fitness[best_idx]

# ----------------------------------------------------
# RUN OPTIMIZATION
# ----------------------------------------------------
board = [
    ["X", " ", " "],
    [" ", "O", " "],
    [" ", " ", " "]
]

best_sequence, best_score = cuckoo_search_best_sequence(board)

print("BEST MOVE SEQUENCE (AI O):")
print(best_sequence)
print("Fitness Score:", best_score)
