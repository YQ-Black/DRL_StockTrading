import torch
from torch import nn
import torch.nn.functional as Func
from ReplayBuffer import ReplayBuffer

numberOfNeurons = 512
dropout = 0.2
learning_rate = 0.01


class DQN(torch.nn.Module):
    def __init__(self, numberOfInputs, numberOfOutputs, dropout=dropout, numberOfNeurons=numberOfNeurons):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(numberOfInputs, numberOfNeurons)
        self.fc2 = nn.Linear(numberOfNeurons, numberOfNeurons)
        self.fc3 = nn.Linear(numberOfNeurons, numberOfNeurons)
        self.fc4 = nn.Linear(numberOfNeurons, numberOfNeurons)
        self.fc5 = nn.Linear(numberOfNeurons, numberOfOutputs)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

    def forward(self, stockData):
        # mid = self.fc1(stockData)
        mid = self.dropout1(Func.relu(self.fc1(stockData)))
        mid = self.dropout2(Func.relu(self.fc2(mid)))
        mid = self.dropout3(Func.relu(self.fc3(mid)))
        mid = self.dropout4(Func.relu(self.fc4(mid)))
        result = self.fc5(mid)

        return result


class Agent:
    '''
    一个DQN的agent，例化之后可以实现以下功能:

    1.stateProcess：对环境输出的state/observation做处理
    2.choose_action: 根据state做出最优的action
    3.store_exp: 把experience存入buffer
    4.QValue: 找出最大的q value
    5.batch_learning: 从replay buffer中随机学习
    6.step_learning: 对整个股票序列数据做顺序学习
    7.

    '''

    def __init__(self, input_features, lossfunc="MSE", optimization="Adam", learningrate=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = DQN(numberOfInputs=input_features, numberOfNeurons=256, numberOfOutputs=3)
        self.network = self.network.cuda()
        self.parameters = self.network.parameters()
        self.learningRate = learningrate
        self.buffer = ReplayBuffer()

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

    def stateProcess(self, state) -> list:
        '''
        把环境反馈出来的observation（state）做处理，以便后续变成正确形状的张量传入神经网络

        :param state: 环境反馈的state
        :return: 处理过后的state
        '''
        temp_0 = state[0]
        temp_1 = state[1]
        state = temp_0 + temp_1
        return state

    def choose_action(self, state, device) -> tuple:
        '''
        根据当前的state判断哪个action的q_value最大

        :param state: 经过process之后的状态
        :param device: 操作使用的设备，cpu/cuda
        :return: q值最大的action序号: -1, 0, 1(int)； q_value_max: tensor(tensor(1.2957, device='cuda:0'))
        '''
        with torch.no_grad():
            # unsqueeze：给生成的tensor多一个维度，例如原来维度是[2,7], 现在维度是[1,2,7]
            state = torch.tensor(state, dtype=torch.float32,
                                 device=device, requires_grad=True).unsqueeze(0)
            # tensor([[  149.5674,   167.7487, 15643.1562, 15354.1562]], device='cuda:0',
            #        requires_grad=True) torch.Size([1, x]) x=特征数
            # print(state, state.shape)
            q_value = self.network(state)
            action = q_value.argmax(1).item() - 1
            q_value_max = q_value.argmax(1).item()
            q_value_max = q_value[0, q_value_max]
            return q_value_max, action

    def store_exp(self, experience: tuple) -> int:
        '''

        :param experience: 一条experience, 包含state，action，reward，state_next
        :return: 此时replaybuffer中指针index的位置
        '''
        index = self.buffer.push(experience)
        return index

    def QValue(self, state: list):
        '''

        :param state: 观察到的状态
        :return: 最大的Q值: tensor(tensor(1.2957, device='cuda:0')
        '''
        state = torch.tensor(state, dtype=torch.float32,
                             device=self.device, requires_grad=True).unsqueeze(0)
        q_value = self.network(state)
        q_value_max = q_value.argmax(1).item()
        q_value_max = q_value[0, q_value_max]
        return q_value_max

    def batch_learning(self, batch_size, env, save=True):
        '''

        :param batch_size: 一次从buffer中抽取的数据数量
        :param env: 交互的环境
        :param save: 是否保存训练的模型
        :return: 啥也没有
        '''
        states, actions, rewards, states_future = self.buffer.random_sample(batch_size)
        loss = 0
        for i in range(batch_size):
            state = states[i]
            action = actions[i]
            reward = rewards[i]
            state_next = states_future[i]
            current_q_value = self.QValue(state)
            next_q_value = self.QValue(state_next)
            target_q_value = reward + env.gamma * next_q_value
            loss = loss + self.loss_fn(current_q_value, target_q_value)

        loss = loss / batch_size
        loss.backward()
        self.optim.step()

        if save == True:
            print("Step training done! Saving model......\n")
            torch.save(self.network, 'dqn_test.pth')
        else:
            print("Step training done!\n")
            return 0

        return 0

    def step_learning(self, env, save=True, steps=1, load=False, load_buffer=False):
        '''
        学习step次直到terminated或者truncated，保存模型

        :param env: 需要与agent交互的环境
        :param save: 是否保存模型
        :param steps: 总共训练多少个step
        :param load: 是否加载模型，目前没有训练
        :param load_buffer: 是否把这次的结果存入replay buffer
        :return: 啥也不返回
        '''
        device = self.device
        observation, info = env.reset()
        observation = self.stateProcess(observation)
        # 暂且不动info了。info = torch.tensor(info, device=self.device)
        current_q_value, action = self.choose_action(observation, device=device)
        current_q_value.requires_grad_(True)
        terminated = 0
        truncated = 0
        for i in range(steps):
            print(f"------第{i + 1}轮step learning开始------")
            j = 0
            while (not terminated) and (not truncated):
                obs, reward, terminated, truncated, info = env.step(action)
                obs = self.stateProcess(obs)
                next_q_value, action = self.choose_action(obs, device)
                target_q_value = reward + (env.gamma * next_q_value)
                target_q_value.requires_grad_(True)
                print("Current q value: ", current_q_value)
                print("Target q value: ", target_q_value)
                loss = self.loss_fn(current_q_value, target_q_value)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                j = j + 1
            else:
                observation, info = env.reset()
                observation = self.stateProcess(observation)
                current_q_value, action = self.choose_action(observation, device=device)
                current_q_value.requires_grad_(True)
                terminated = 0
                truncated = 0

            print(f"本轮总共训练了{j}次")

        if save == True:
            print("Step training done! Saving model......\n")
            torch.save(self.network, 'dqn_test.pth')
        else:
            print("Step training done!\n")
            return 0
