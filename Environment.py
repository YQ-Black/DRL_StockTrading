import gymnasium
from gymnasium import spaces
from Data_preprocess import CSVProcessor
import numpy as np

class StockTradingEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, dataPath, money, startDate, tradingCosts=0, obsPeriod=1):
        '''
        目的：初始化环境，并配置相应参数

        :param dataPath: 数据的文件路径
        :param money: 开始拥有的本金数量
        :param startDate: 开始时间(应该观察期之后才能开始)
        :param tradingCosts: 交易成本，一般是百分之几（0.01---1%）
        :param obsPeriod: 观察状态的长度。（e.g. 1表示1天，7表示7天）
        '''
        # 基本量的设置
        self.env_name = 'StockTradingEnv'
        self.max_step = 200 #暂定
        self.money = money
        self.startDate = startDate
        self.tradingCosts = tradingCosts
        self.done = False
        self.obsPeriod = obsPeriod
        self.t = self.obsPeriod

        # 数据的处理，以dataframe形式保存
        dataprocess = CSVProcessor()
        stockData = dataprocess.data_pre_process(dataPath)
        self.stockData = stockData

        # gym库的设置
        self.state_dim = spaces.Discrete(2)  # six observations: open, Close, high, low, volumn, revenue。目前暂定open close
        self.action_dim = spaces.Discrete(3) # We have 3 actions, corresponding to "buy", "hold", "sell"(1, 0, -1)
        self.if_discrete = True

    def _get_obs(self, index:int)->list:
        return [self.stockData['Open'][index:index+self.obsPeriod].tolist(),
                self.stockData['Close'][index:index+self.obsPeriod].tolist()]

    def _get_info(self, index:int)->int:
        return self.stockData['Volume'][index]


    def reset(self, seed=None, options=None)->tuple:
        super().reset(seed=seed)
        # 对基本量的设置
        self.Return = 0
        self.done = False
        self.endDate = self.stockData.loc[self.stockData.index[-1], 'Date']
        self.nowDate = self.startDate # self.nowIndex = 0 下面observation直接变成[nowIndex:nowIndex + self.obsPeriod](which is better?)
        self.stockOwned = 0
        self.stockOwnedHis = 0

        # observation暂定为Open和Close两个特征，总共有obsPeriod个。Return type: list
        observation = self._get_obs(self.nowDate)
        info = self._get_info(self.nowDate)
        return observation, info

    def step(self, action):
        t = self.t
        # self.stockOwnedHis = self.stockOwned
        # return state ,reward, done, info


    def _take_action(self, action):
        if action == -1:
            price_today = self.stockData
            


