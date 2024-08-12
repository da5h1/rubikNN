from cube import Rubik
from drive import Drive
from utils import square_to_index, get_move_to_index
import json
import numpy as np
import random
import torch
from torch import nn
with open("data/face_map.json") as f:
    face_map = json.load(f)
num_iterations = 100
move_to_index = get_move_to_index()
index_to_move = {v: k for k, v in move_to_index.items()}
solved_number = 0
solver_mode = True
print("solver mode", solver_mode)
if solver_mode:
    num_iterations = 1

init_config = {'w': ['w9', 'r8', 'g9', 'w8', 'w5', 'g8', 'r1', 'b8', 'y9'], 'r': ['b9', 'r4', 'r7', 'b6', 'r5', 'y4', 'o3', 'w2', 'y7'], 'g': ['o9', 'g6', 'b3', 'o6', 'g5', 'r6', 'b7', 'g4', 'r9'], 'o': ['b1', 'o4', 'o7', 'b4', 'o5', 'w6', 'r3', 'y8', 'w3'], 'b': ['w7', 'w4', 'g7', 'r2', 'b5', 'o2', 'g3', 'y6', 'y3'], 'y': ['w1', 'b2', 'o1', 'g2', 'y5', 'y2', 'g1', 'o8', 'y1']}

class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.model = nn.Sequential(
        nn.Linear(54*6, 8192),
        nn.ReLU(inplace = True),
        nn.Linear(8192, 4096),
        nn.ReLU(inplace = True),
        nn.Linear(4096, 2048),
        nn.ReLU(inplace = True),
        nn.Linear(2048, 512),
        nn.ReLU(inplace = True),
        nn.Linear(512, 18)
    )

  def forward(self, x):
    x = x.view(x.shape[0], 54*6)
    x = self.model(x)
    return x

model = Model()
model.load_state_dict(torch.load("/userspace/cdd/rubik/training/model.pth", map_location=torch.device('cpu')))
def get_move_from_model(model: nn.Module, t: torch.Tensor):
    outputs = model(t)
    outputs = outputs.detach().cpu().numpy() if outputs.requires_grad else outputs.cpu().numpy()
    y_pred = np.argmax(outputs, axis = -1)[0]
    move = index_to_move[y_pred]
    return move

for i in range(num_iterations):
    cube = Rubik()
    mover = Drive(face_map)
    if solver_mode:
        cube.update(init_config)
    else:
        moves1 = mover.parser(mover.random_moves(17))#
        for move in moves1:
            new_conf = mover.move(cube.get_conf(), move)
            cube.update(new_conf)
        
    moves2 = []
    counter = 0
    while not cube.is_solved():
        counter += 1
        t = torch.Tensor(cube.get_arr(one_hot = True)).unsqueeze_(0)
        move = get_move_from_model(model, t)
        moves2.append(move)
        new_conf = mover.move(cube.get_conf(), move)
        cube.update(new_conf)
        if counter == 150:
            break
        
    if cube.is_solved():
        if not solver_mode:
            print(moves1)
        print(moves2)
        solved_number += 1

print(f"Solved: {solved_number/num_iterations}")
        
        
        
        