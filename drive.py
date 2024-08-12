import json
import random
import copy

class Drive:
    def __init__(self, face_map: dict):
        self.dcube = None
        self.face_map = face_map
    
    def move(self, dcube: dict, inpt: str) -> dict:
        moves = self.parser(inpt)
        self.dcube = dcube
        for move in moves:
            self.apply(move)
        return self.dcube
        
    def random_move(self) -> str:
        moves = ['f', 'b', 'u', 'd', 'r', 'l']
        rand_move = str(moves[random.randint(0, 5)])
        rand_int = str(random.randint(1, 3))
        return rand_move + rand_int
    
    def random_moves(self, n: int) -> str:
        s = ''
        i = 0
        while i < n:
            move = self.random_move()
            if i > 0 and move[0] == s[-2]:
                continue
            s += move
            i += 1
        return s
            
    def parser(self, text: str) -> list:
        moves = [text[i] + text[i + 1] for i in range(0, len(text), 2)]
        return moves
    
    def apply(self, move: str):
        if move[0] == 'f':
            for i in range(int(move[1])):
                self.move_face('w')
        elif move[0] == 'u':
            for i in range(int(move[1])):
                self.move_face('r')
        elif move[0] == 'd':
            for i in range(int(move[1])):
                self.move_face('o')
        elif move[0] == 'l':
            for i in range(int(move[1])):
                self.move_face('b')
        elif move[0] == 'r':
            for i in range(int(move[1])):
                self.move_face('g')
        elif move[0] == 'b':
            for i in range(int(move[1])):
                self.move_face('y')
    
    def move_face(self, face: str):
        new_dcube = copy.deepcopy(self.dcube)
        mface = self.face_map[face]
        new_dcube[face][0] = self.dcube[face][6]
        new_dcube[face][1] = self.dcube[face][3]
        new_dcube[face][2] = self.dcube[face][0]
        new_dcube[face][3] = self.dcube[face][7]
        new_dcube[face][5] = self.dcube[face][1]
        new_dcube[face][6] = self.dcube[face][8]
        new_dcube[face][7] = self.dcube[face][5]
        new_dcube[face][8] = self.dcube[face][2]

        color_a, color_b = (list(mface.keys())[0], list(mface.keys())[-1])
        for a, b in zip(mface[color_a], mface[color_b]):
            new_dcube[color_a][a] = self.dcube[color_b][b]
            
        color_a, color_b = (list(mface.keys())[1], list(mface.keys())[0])
        for a, b in zip(mface[color_a], mface[color_b]):
            new_dcube[color_a][a] = self.dcube[color_b][b]

        color_a, color_b = (list(mface.keys())[2], list(mface.keys())[1])
        for a, b in zip(mface[color_a], mface[color_b]):
            new_dcube[color_a][a] = self.dcube[color_b][b]

        color_a, color_b = (list(mface.keys())[3], list(mface.keys())[2])
        for a, b in zip(mface[color_a], mface[color_b]):
            new_dcube[color_a][a] = self.dcube[color_b][b]

        self.dcube = new_dcube
        return self
    
        

        
        
        