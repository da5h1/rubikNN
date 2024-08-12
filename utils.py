import numpy as np

def get_move_to_index() -> dict:
    move_to_index = {}
    i = 0
    for move in 'fbudrl':
        for k in range(1,4):
            move_to_index[move+str(k)] = i
            i += 1
    return move_to_index

def square_to_index(s: str, one_hot: bool) -> dict:
    idx = 'wrgoby'.index(s[0])
    if not one_hot:
        return idx
    else:
        return np.eye(6)[idx]

def reverse_move(move: str):
    s = ''
    s += move[0]
    if move[1] == '1':
        s += '3'
    elif move[1] == '3':
        s += '1'
    else:
        s += move[1]
    return s
    
    