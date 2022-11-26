import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(device)

