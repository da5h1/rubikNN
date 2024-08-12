from cube import Rubik
from drive import Drive
from utils import square_to_index, get_move_to_index, reverse_move
import json
import numpy as np
import random
import tqdm
import torch

with open("data/face_map.json") as f:
    face_map = json.load(f)

dataset_type = "train"
num_iterations = 1000000
mover = Drive(face_map)

move_to_index = get_move_to_index()
X = []
Y = []

for _ in tqdm.tqdm(range(num_iterations)):
    cube = Rubik()
    moves = mover.parser(mover.random_moves(20))
    for move in moves:
        Y.append(move_to_index[reverse_move(move)])
        new_conf = mover.move(cube.get_conf(), move)
        cube.update(new_conf)
        X.append(cube.get_arr(one_hot=True))

X = np.array(X)
Y = np.array(Y)

np.save(f"./data/X{dataset_type}.npy", X)
np.save(f"./data/Y{dataset_type}.npy", Y)

'''
train_ds = [(torch.LongTensor(X[i]), Y[i]) for i in range(len(X))]

def tensor_to_config(t: torch.Tensor) -> str:
  output = ""
  s = "wrgoby"
  for face in t:
    for row in face:
      for x in row:
        index = np.argmax(x)
        output += s[index]
  return output

def idx2move(idx: int) -> str:
  m2i = get_move_to_index()
  i2m = {v: k for k, v in m2i.items()}
  return i2m[idx]


for i, x in enumerate(train_ds):
  print(tensor_to_config(x[0]))
  print(idx2move(x[1].item()))
  if i == 200:
    break
'''

