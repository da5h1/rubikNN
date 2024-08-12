from utils import square_to_index
import numpy as np

class Rubik:
    def __init__(self):
        self.solved = {
            "w": ["w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9"],
            "r": ["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9"],
            "g": ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9"],
            "o": ["o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8", "o9"],
            "b": ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9"],
            "y": ["y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8", "y9"]
        }
        self.dcube = self.solved
    
    def get_conf(self) -> dict:
        return self.dcube
    
    def update(self, cube: dict):
        self.dcube = cube
    
    def get_arr(self, one_hot = False):
        result = [[] for _ in range(6)]
        pairs = list(self.dcube.items())
        for i,x in enumerate(pairs):
            result[i].append([square_to_index(y, one_hot) for y in x[1][0:3]])
            result[i].append([square_to_index(y, one_hot) for y in x[1][3:6]])
            result[i].append([square_to_index(y, one_hot) for y in x[1][6:9]])
        return np.array(result)
    
    def is_solved(self):
        return self.dcube == self.solved
    
        
        