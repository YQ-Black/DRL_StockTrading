import torch
from torch import nn
import torch.nn.functional as Func
from ReplayBuffer import ReplayBuffer

numberOfNeurons = 512
dropout = 0.2
learning_rate = 0.01
dataset_divider = 3273  # 用于分割训练集与测试集。可灵活调整


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
    """
    一个DQN的agent，例化之后可以实现以下功能:

    1.stateProcess：对环境输出的state/observation做处理\n
    2.choose_action: 根据state做出最优的action\n
    3.store_exp: 把experience存入buffer\n
    4.QValues: 找出最大的q value\n
    5.batch_learning: 从replay buffer中随机学习\n
    6.step_learning: 对整个股票序列数据做顺序学习\n

    """

    def __init__(self, input_features: int, env, lossfunc="MSE", optimization="Adam", learningrate=0.01):
        """
        初始化Agent类，用于后续其他操作。

        :param input_features: 取决于obsPeriod和observation space的大小。  input features=（obsPeriod+1）*obs_space
        :param lossfunc: 目前提供：MSE, L1Loss, C-E Loss
        :param optimization: 目前提供：Adam, SGD
        :param learningrate: 学习率，default=0.01
        """
        # 初始化所有的character
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = DQN(numberOfInputs=input_features, numberOfNeurons=256, numberOfOutputs=3)
        self.network = self.network.cuda()
        self.target_network = DQN(numberOfInputs=input_features, numberOfNeurons=256, numberOfOutputs=3)
        self.target_network = self.target_network.cuda()
        self.parameters = self.network.parameters()
        self.learningRate = learningrate
        self.Buffer = ReplayBuffer()
        self.env = env

        # 复制network的参数给target network
        network_paras = self.network.state_dict()
        self.target_network.load_state_dict(network_paras)

        # 选择loss function 和 optimizer
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

    def stateProcess(self, state: list) -> list:
        """
        把环境反馈出来的observation（state）做处理，以便后续变成正确形状的张量传入神经网络

        :param state: 环境反馈的state
        :return: 处理过后的state
        """
        temp_0 = state[0]
        temp_1 = state[1]
        state = temp_0 + temp_1
        return state

    def choose_action(self, state: list, device, network) -> tuple:
        """
        根据当前的state判断哪个action的q_value最大

        :param state: 经过process之后的状态
        :param device: 操作使用的设备，cpu/cuda
        :return: ①q_value_max: tensor(tensor(1.2957, device='cuda:0')) ②q值最大的action序号: -1, 0, 1(int)
        """
        if network == self.network:
            with torch.no_grad():
                # unsqueeze：给生成的tensor多一个维度，例如原来维度是[2,7], 现在维度是[1,2,7]
                state = torch.tensor(state, dtype=torch.float32,
                                     device=device, requires_grad=True).unsqueeze(0)
                # tensor([[  149.5674,   167.7487, 15643.1562, 15354.1562]], device='cuda:0',
                #        requires_grad=True) torch.Size([1, x]) x=特征数
                # print(state, state.shape)
                q_value = self.network(state)
                print(q_value, type(q_value))
                action = q_value.argmax(1).item() - 1
                q_value_max = q_value.argmax(1).item()
                # print(q_value, type(q_value))
                q_value_max = q_value[0, q_value_max]
                return q_value_max, action

        elif network == self.target_network:
            with torch.no_grad():
                # unsqueeze：给生成的tensor多一个维度，例如原来维度是[2,7], 现在维度是[1,2,7]
                state = torch.tensor(state, dtype=torch.float32,
                                     device=device, requires_grad=True).unsqueeze(0)
                # tensor([[  149.5674,   167.7487, 15643.1562, 15354.1562]], device='cuda:0',
                #        requires_grad=True) torch.Size([1, x]) x=特征数
                # print(state, state.shape)
                q_value = self.target_network(state)
                print(q_value, type(q_value))
                action = q_value.argmax(1).item() - 1
                q_value_max = q_value.argmax(1).item()
                # print(q_value, type(q_value))
                q_value_max = q_value[0, q_value_max]
                return q_value_max, action
        else:
            print("the network you choose is wrong\n"
                  "please choose target_network or network!\n")
            return -1, -1


    def QValues(self, states: list, network, batch_size=1):
        """
        根据一系列输入状态得到一系列的一组Qmax，用于batch learning函数

        :param batch_size: 观察到的一系列状态(batch size)
        :param states: 观察到的一系列状态(batch size)
        :return: 最大的一组Q值: shape: torch.Size([1, batch_size])
        """
        # states = torch.tensor(states, dtype=torch.float32,
        #                      device=self.device, requires_grad=True).unsqueeze(0)

        if network == self.network:
            q_values = self.network(states)

            q_values_list = []

            for i in range(batch_size):
                q_value = q_values[0][i]
                q_value = q_value.unsqueeze(0)
                q_index = q_value.argmax(1).item()
                q_value_max = q_value[0, q_index]
                q_value_max = [q_value_max]
                q_values_list = q_values_list + q_value_max

            q_values_max = torch.tensor(q_values_list, dtype=torch.float32,
                                        device=self.device, requires_grad=True).unsqueeze(0)

            return q_values_max

        elif network == self.target_network:
            q_values = self.target_network(states)

            q_values_list = []

            for i in range(batch_size):
                q_value = q_values[0][i]
                q_value = q_value.unsqueeze(0)
                q_index = q_value.argmax(1).item()
                q_value_max = q_value[0, q_index]
                q_value_max = [q_value_max]
                q_values_list = q_values_list + q_value_max

            q_values_max = torch.tensor(q_values_list, dtype=torch.float32,
                                        device=self.device, requires_grad=True).unsqueeze(0)

            return q_values_max

        else:
            print("the network you choose is wrong\n"
                  "please choose target_network or network!\n")
            return -1

    def store_exp(self) -> int:
        """
        目前只能读取环境内一支股票的数据，也就是只有3000多天的数据

        :return: 此时replay buffer中指针index的位置
        """
        device = self.device
        observation, info = self.env.reset()
        observation = self.stateProcess(observation)
        current_q_value, action = self.choose_action(observation, device=device)
        current_q_value.requires_grad_(True)
        terminated = False
        truncated = False

        print("------存储experience开始------")
        j = 0
        while (not terminated) and (not truncated):
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            next_obs = self.stateProcess(next_obs)
            exp = self.Buffer.exp_capsu(observation, action, reward, next_obs)
            self.Buffer.push(exp)

            # 把上面得到的状态S_{t+1}变为S_{t}
            observation = next_obs
            current_q_value, action = self.choose_action(observation, device=device)
            current_q_value.requires_grad_(True)

            j = j + 1
            # print(f"存储第{j}条experience")

        print(f"experience存储完毕，总共存储了{j}条experience")
        print("store_exp done!\n")

        return 0

    def batch_learning(self, batch_size: int, env, save=True) -> int:
        """
        从replay buffer中一次抽取batch size个数据进行训练。

        :param batch_size: 一次从buffer中抽取的数据数量。不能超过数据总量
        :param env: 交互的环境
        :param save: 是否保存训练的模型
        :return: 啥也没有
        """
        states, actions, rewards, states_future = self.Buffer.random_sample(batch_size, self.device)
        # 假设batch_size=16; 两个特征,一个特征看6天（2*6=12）. states.shape: torch.Size([16, 12])

        current_q_values = self.QValues(network=self.network, states=states, batch_size=batch_size)

        next_q_values = self.QValues(network=self.target_network, states=states, batch_size=batch_size)
        target_q_values = rewards + env.gamma * next_q_values

        loss = self.loss_fn(current_q_values, target_q_values)

        loss.backward()
        self.optim.step()

        if save:
            print("Batch training done! Saving model......\n")
            torch.save(self.network, 'dqn_test.pth')
            return 0
        else:
            # print("Batch training done!\n")
            return 0

    def store_experience_n_train(self, env, save=True, steps=1, load=False, load_buffer=False) -> int:
        """
        顺序学习股票数据，在放入buffer的同时小小训练一下网络

        :param env: 需要与agent交互的环境
        :param save: 是否保存模型
        :param steps: 总共训练多少个step
        :param load: 是否加载模型，目前没有训练
        :param load_buffer: 是否把这次的结果存入replay buffer
        :return: 0
        """
        device = self.device
        observation, info = env.reset()
        observation = self.stateProcess(observation)
        # 暂且不动info了。info = torch.tensor(info, device=self.device)
        current_q_value, action = self.choose_action(observation, device=device, network=self.network)
        current_q_value.requires_grad_(True)
        terminated = False
        truncated = False

        for i in range(steps):
            print(f"------第{i + 1}轮step learning开始------")
            while (not terminated) and (not truncated):
                next_obs, reward, terminated, truncated, info = env.step(action)
                next_obs = self.stateProcess(next_obs)

                exp = self.Buffer.exp_capsu(observation, action, reward, next_obs)
                self.Buffer.push(exp)

                next_q_value, action = self.choose_action(next_obs, device, network=self.target_network)
                target_q_value = reward + (env.gamma * next_q_value)
                target_q_value.requires_grad_(True)
                print("Current q value: ", current_q_value)
                print("Target q value: ", target_q_value)

                loss = self.loss_fn(current_q_value, target_q_value)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                current_q_value, action = self.choose_action(next_obs, device=device, network=self.network)

        if save:
            print("Step training done! Saving model......\n")
            torch.save(self.network, 'dqn_test.pth')
            return 0
        else:
            print("------ Experience done! ------\n")
            return 0

    def test_reset(self) -> None:
        """
        重新设置本金为10000，把日期调整到2023年的第一个数据上

        :return: None
        """
        self.env.asset = 10000
        self.env.stockHold = 0
        self.env.cash = 10000
        self.env.nowDate = dataset_divider

    def test_model(self) -> None:
        """
        环境会自动回到2023年的第一天，然后开始对2023年的所有股票数据进行测试,并记录到dataframe中，用于后续绘制图像

        :return: None
        """
        terminated = False
        truncated = False
        self.test_reset()

        observation = self.env._get_obs(self.env.nowDate)
        observation = self.stateProcess(observation)

        current_q_value, action = self.choose_action(observation, device=self.device, network=self.network)
        current_q_value.requires_grad_(True)

        while (not terminated) and (not truncated):
            obs, reward, terminated, truncated, info = self.env.step(action)
            obs = self.stateProcess(obs)
            next_q_value, action = self.choose_action(obs, self.device, network=self.network)

    def train_n_test(self, epoch, episode) -> None:
        """
        将训练集中的数据全部存入replay buffer中并进行训练和测试，绘制本次测试的结果，然后保存模型

        :param epoch: 外层大循环。总共进行多少次训练
        :param episode: 内层小循环。从replay buffer中取出多少个batch来训练
        :return:
        """
        self.store_exp()

        for i in range(epoch):
            print(f"------第{i}个epoch train开始------")
            for j in range(episode):
                self.batch_learning(batch_size=64, env=self.env, save=False)

            parameters = self.network.state_dict()
            self.target_network.load_state_dict(parameters) # update target network

            print(f"------第{i}个epoch test开始------")
            self.test_model()
            self.env.render(epoch=i)
            torch.save(self.network, 'DQN_{}.pth'.format(i))
