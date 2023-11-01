import torch
from torch import nn
import torch.nn.functional as Func

numberOfNeurons = 512
dropout = 0.2
learning_rate = 0.01

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
        mid = self.fc1(stockData)
        #mid = self.dropout1(Func.relu(self.bn1(self.fc1(stockData))))
        mid = self.dropout2(Func.relu(self.bn2(self.fc2(mid))))
        mid = self.dropout3(Func.relu(self.bn3(self.fc3(mid))))
        mid = self.dropout4(Func.relu(self.bn4(self.fc4(mid))))
        result = self.fc5(mid)

        return result

    def training_sarsa(self, Env, Lossfunc, Optim):
        stockdata = Env.stockDate

class Agent():
    def __init__(self, lossfunc="MSE", optimization="Adam", gamma=0.99, learningrate=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = DQN(numberOfInputs=2, numberOfNeurons=256, numberOfOutputs=3)
        self.parameters = self.network.parameters()
        self.gamma = gamma
        self.learningRate = learningrate

        Lossfunc = {"MSE": nn.MSELoss(), "L1Loss": nn.L1Loss(), "C-E Loss": nn.CrossEntropyLoss()}
        if (lossfunc in Lossfunc):
            loss_fn = Lossfunc[lossfunc]
            self.loss_fn = loss_fn.cuda()
        else:
            print("No such loss function, please input again!\n")

        if optimization == "Adam":
            self.optim = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        elif optimization == "SGD":
            self.optim = torch.optim.SGD(self.network.parameters(), lr=learning_rate)
        else:
            print("No such optimization, please input again!\n")

    def choose_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_value = self.network(state)
            return q_value.argmax(1).item()


    def learning(self, env, save=False, epoch=1, load=False):
        observation, info = env.reset()
        observation = torch.tensor(observation)
        info = torch.tensor(info)
        action = self.choose_action(observation)
        for i in range(epoch):
            print(f"------第{i+1}轮训练开始------")
            obs, reward, terminated, truncated, info = env.step(action)



