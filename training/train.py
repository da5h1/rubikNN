# -*- coding: utf-8 -*-
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import sys
sys.path.append("..")
from utils import get_move_to_index
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('device:', device)

t0 = time.time()
num_epochs = 25
print('NEPOCHS:', num_epochs)
batch_size = 128
print('batch size:', batch_size)

X_train = np.load("/userspace/cdd/rubik/data/Xtrain.npy")
Y_train = np.load("/userspace/cdd/rubik/data/Ytrain.npy")
X_val = np.load("/userspace/cdd/rubik/data/Xval.npy")
Y_val = np.load("/userspace/cdd/rubik/data/Yval.npy")

class RubikDataset(Dataset):
  def __init__(self, X: np.ndarray, Y: np.ndarray):
    self.X = torch.Tensor(X)
    self.Y = torch.LongTensor(Y)

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx: int):
    return self.X[idx], self.Y[idx]

train_ds = RubikDataset(X_train, Y_train)
train_loader = DataLoader(train_ds, batch_size = batch_size,
                    drop_last = True, shuffle = True)
print(train_ds.X.size(0), "samples in train_ds")

val_ds = RubikDataset(X_val, Y_val)
val_loader = DataLoader(val_ds, batch_size = batch_size,
                        drop_last = True, shuffle = True)
print(val_ds.X.size(0), "samples in val_ds")

class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.model = nn.Sequential(
        nn.Linear(6*6*9, 8192),
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
    x = x.view(x.shape[0], 6*6*9)
    x = self.model(x)
    return x

model = Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

def get_score(outputs: torch.Tensor, y: torch.Tensor, verbose = False) -> np.float64:
  outputs = outputs.detach().cpu().numpy() if outputs.requires_grad else outputs.cpu().numpy()
  y_pred = np.argmax(outputs, axis = -1)
  y_true = y.detach().cpu().numpy() if y.requires_grad else y.cpu().numpy()
  return accuracy_score(y_true, y_pred)

def train_model(model: nn.Module, criterion: callable,
                optimizer: torch.optim.Optimizer, train_loader: DataLoader,
                val_loader: DataLoader, num_epochs: int):
  best_val_loss = float('inf')
  for epoch in range(num_epochs):
    train_losses = []
    train_scores = []
    for x, y in train_loader:
      model.train()
      x = x.to(device)
      y = y.to(device).view(batch_size)
      optimizer.zero_grad()
      outputs = model(x)
      train_loss = criterion(outputs, y)
      train_score = get_score(outputs, y)
      train_loss.backward()
      optimizer.step()
      train_losses.append(train_loss.item())
      train_scores.append(train_score)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {np.mean(train_losses):.4f}, Train Score: {np.mean(train_scores):.4f}")
    if epoch % 5 == 0:
      model.eval()
      val_losses = []
      val_scores = []
      with torch.no_grad():
        for x, y in val_loader:
          x = x.to(device)
          y = y.to(device).view(batch_size)
          outputs = model(x)
          val_loss = criterion(outputs, y)
          val_score = get_score(outputs, y)
          val_losses.append(val_loss.item())
          val_scores.append(val_score)
      print(f"Val Loss: {np.mean(val_losses):.4f}, Val Score: {np.mean(val_scores):.4f}")
         
      val_loss = np.mean(val_losses)
      if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "model.pth")
    
start_time = time.time()
train_model(model = model, criterion = criterion,
            optimizer = optimizer, train_loader = train_loader,
            val_loader = val_loader, num_epochs = num_epochs)

end_time = time.time()
print("training time", end_time - start_time)


t1 = time.time()

print(t1-t0)

