import os.path

import gymnasium
from gymnasium import spaces
from Data_preprocess import CSVProcessor
import numpy as np
import math
import matplotlib.pyplot as plt

class StockTradingEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, money, startDate=0, tradingCosts=0, obsPeriod=1, company="Apple"):
        '''
        目的：初始化环境，并配置相应参数

        :param company: 用于选择公司，以之后加载相应数据。目前提供：Apple, Tesla, Amazon, AMD, NVIDIA, Microsoft, Intel, Google
        :param money: 开始拥有的本金数量
        :param startDate: 开始时间(应该观察期之后才能开始)
        :param tradingCosts: 交易成本，一般是百分之几（0.01---1%）
        :param obsPeriod: 观察状态的长度。（e.g. 1表示1天，7表示7天）
        '''
        # 基本量的设置
        self.Company = {"Apple":"AAPL", "Tesla":"TSLA", "Amazon":"AMZN", "AMD":"AMD",
                        "NVIDIA":"NVDA", "Microsoft":"MSFT", "Intel":"INTC", "Google":"GOOG"}
        self.com_name = company
        self.env_name = 'StockTradingEnv'
        self.max_step = 250 #暂定
        self.asset = money
        self.startDate = startDate
        self.tradingCosts = tradingCosts
        self.done = False
        self.obsPeriod = obsPeriod
        # self.t = self.obsPeriod
        self.stockHold = 0
        self.cash = money
        self.assetHis = 0
        self.threshold = 0.1*money

        # gym库的设置
        self.state_dim = spaces.Discrete(2)  # six observations: open, Close, high, low, volumn, revenue。目前暂定open close
        self.action_dim = spaces.Discrete(3)  # We have 3 actions, corresponding to "buy", "hold", "sell"(1, 0, -1)
        self.if_discrete = True

        # 数据的处理，以dataframe形式保存
        file = self.Company[company] + ".csv"
        filepath = "D:\science\Design\Data"
        file = os.path.join(filepath, file)

        if os.path.isfile(file):
            dataprocess = CSVProcessor()
            stockData = dataprocess.data_pre_process(file)
            self.stockData = stockData

            # 新添加几列别的属性
            self.stockData["Stock Hold"] = 0.
            self.stockData["Cash Hold"]  = 0.
            self.stockData["Asset"] = 0.
        else:
            print("Error! Initialization failed. Reason: No data found, please download data first.\n")




    def _get_obs(self, index:int)->list:
        '''
        Return observations as Open and Close data

        :param index: observation从哪一天开始
        :return: list类型
        '''
        return [self.stockData['Open'][index:index+self.obsPeriod].tolist(),
                self.stockData['Close'][index:index+self.obsPeriod].tolist()]

    def _get_info(self, index:int)->int:
        '''
        返回info，暂定为股票当日的volume

        :param index: 哪一天的数据
        :return: int类型
        '''
        return self.stockData['Volume'][index]
        # 先列后行


    def reset(self, seed=None, options=None)->tuple:
        super().reset(seed=seed)
        # 对基本量的设置
        self.Return = 0
        self.step_cnt = 0
        self.endDate = self.max_step
        self.nowDate = self.startDate # self.nowIndex = 0 下面observation直接变成[nowIndex:nowIndex + self.obsPeriod](which is better?)

        # observation暂定为Open和Close两个特征，总共有obsPeriod个。Return type: list
        # info暂定为Volume这一个特征，总共一个，当日的volume。Return type: int
        # 先行后列
        t = self.startDate
        self.stockData.loc[t, "Stock Hold"] = self.stockHold
        self.stockData.loc[t, "Cash Hold"]  = self.cash
        self.stockData.loc[t, "Asset"]      = self.asset

        observation = self._get_obs(self.nowDate)
        info = self._get_info(self.nowDate)
        return observation, info

    def step(self, action)->tuple:
        self._take_action(action)

        self.nowDate = self.nowDate + 1
        t = self.nowDate
        self.stockData.loc[t, "Stock Hold"] = self.stockHold
        self.stockData.loc[t, "Cash Hold"] = self.cash
        self.stockData.loc[t, "Asset"] = self.asset

        observation = self._get_obs(t)
        info = self._get_info(t)
        reward = (self.asset - self.assetHis) / self.assetHis

        terminated = False
        if self.asset < self.threshold or self.step_cnt == self.max_step:
            truncated = True
            self.step_cnt = 0
        else:
            truncated = False
            self.step_cnt = self.step_cnt + 1

        return observation ,reward, terminated, truncated, info


    def _take_action(self, action:int)->int:
        '''
        目的：按照输入的action来更新有关的属性与数值.
        暂定为，根据Open的数据来确定是否buy，hold，sell,在close时结算本日
        :param action: -1:sell, 0:hold, 1:buy
        :return: not known yet
        '''
        today = self.nowDate
        cash = self.cash
        if action == 0:
            self.stockHold = self.stockHold
            self.cash = self.cash
            self.assetHis = self.asset
            self.asset = self.cash + self.stockData['Close'][today] * self.stockHold
            # self.stockData["Stock Hold"] = 0
            # self.stockData["CashHold"] = 0
            # self.stockData["Asset"] = 0

        elif action == 1:
            cost = self.stockData['Open'][today] * (1+self.tradingCosts)
            diff = cash - cost
            if(diff >= 0):
                stockBuy = math.floor(cash/cost)
                self.cash = cash - stockBuy * cost
                self.stockHold = self.stockHold + stockBuy
                self.assetHis = self.asset
                self.asset = self.cash + self.stockData['Close'][today] * self.stockHold
            else:
                self.cash = self.cash
                self.stockHold = self.stockHold
                self.assetHis = self.asset
                self.asset = self.cash + self.stockData['Close'][today] * self.stockHold

        elif action == -1:
            if(self.stockHold >=0):
                revenue = self.stockData['Open'][today] * (1-self.tradingCosts) * self.stockHold
                self.cash = self.cash + revenue
                self.stockHold = 0
                self.assetHis = self.asset
                self.asset = self.cash + self.stockData['Close'][today] * self.stockHold
            else:
                self.cash = self.cash
                self.stockHold = self.stockHold
                self.assetHis = self.asset
                self.asset = self.cash + self.stockData['Close'][today] * self.stockHold

        return 0

    def getAttris(self):
        t = self.nowDate
        print("Today is day ", t, " , Date is ", self.stockData['Date'][t])
        print("Asset: ", self.stockData["Asset"][t], " in day ", t)
        print("Stock Hold: ", self.stockData["Stock Hold"][t], " in day ", t)
        print("Cash Hold: ", self.stockData["Cash Hold"][t], " in day ", t)
        print("Asset yesterday: ", self.stockData["Asset"][t-1], " in day ", t)
        print("------------------------------------")
        print('\n')

    def takeRandAct(self)->int:
        '''
        随机在-1，0，1之间选择一个整数，作为action，用于简单测试
        :return: -1 / 0 / 1
        '''
        act = np.random.randint(-1, 2, dtype=int)
        return act

    def render(self, epoch=None):
        '''
        绘图并按照训练epoch来命名保存保存
        :param epoch: 表示当前是第几个epoch，用于保存图片取名时使用
        :return: 根据数据画出股票数据和拥有资产的折线图
        '''
        df = self.stockData
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
        axes[0].plot(df["Date"], df["Close"])
        axes[0].set_title(f"{self.com_name} Stock Close Price")
        axes[1].plot(df["Asset"])
        axes[1].set_title("Asset Hold")
        if epoch == None:
            plt.savefig(fname="example.jpg")
            plt.show()
        else:
            save_path = f"Figures/fig_{epoch}.jpg"
            plt.savefig(fname=save_path)


    def close(self):
        '''
        这个函数没用，不需要管
        :return: nothing
        '''
        print("Closed Successfully\n")


