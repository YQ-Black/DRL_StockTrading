import torch
from torch import nn
import torch.nn.functional as Func

numberOfNeurons = 512
dropout = 0.2


class DQN(torch.nn.Module):
    def __init__(self, numberOfInputs, numberOfOutputs, dropout=dropout, numberOfNeurons=numberOfNeurons):
        super(DQN,self).__init__()

        self.fc1 = nn.Linear(numberOfInputs,numberOfNeurons)
        self.fc2 = nn.Linear(numberOfNeurons, numberOfNeurons)
        self.fc3 = nn.Linear(numberOfNeurons, numberOfNeurons)
        self.fc4 = nn.Linear(numberOfNeurons, numberOfNeurons)
        self.fc5 = nn.Linear(numberOfNeurons, numberOfOutputs)

        self.bn1 = nn.BatchNorm1d(numberOfNeurons)
        self.bn2 = nn.BatchNorm1d(numberOfNeurons)
        self.bn3 = nn.BatchNorm1d(numberOfNeurons)
        self.bn4 = nn.BatchNorm1d(numberOfNeurons)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

    def forward(self, stockData):
        mid = self.dropout1(Func.relu(self.bn1(self.fc1(stockData))))
        mid = self.dropout2(Func.relu(self.bn2(self.fc2(mid))))
        mid = self.dropout3(Func.relu(self.bn3(self.fc3(mid))))
        mid = self.dropout4(Func.relu(self.bn4(self.fc4(mid))))
        result = self.fc5(mid)

        return result