import random
from torch import tensor

class ReplayBuffer():
    def __init__(self, depth = 10_00000):
        '''
        对replaybuffer做初始化，并确定buffer的深度。

        :param depth: buffer的大小，默认为10e5
        '''
        self.buffer = []
        self.depth = depth
        self.index = 0
        self.flag = 0

    def _flag(self)->int:
        if self.index < self.depth:
            self.flag = self.flag
        else:
            self.flag = 1

        return self.flag

    def exp_capsu(self, state, action, reward, next_state)->tuple:
        experience = (state, action, reward, next_state)
        return experience



    def push(self, experience:tuple)->int:
        '''
        把一个experience放入buffer，并更新buffer指针的位置

        :param experience: 输入必须是tuple形式，否则sample时无法正常工作.experience中的state必须是处理成一维数据，否则无法正常工作
        :return: 返回buffer指针的index
        '''
        index = self.index
        flag = self._flag()
        if (index < self.depth and flag ==0):
            self.buffer.append(experience)
            self.index = self.index + 1
        elif (index < self.depth and flag ==1):
            self.buffer[index] = experience
            self.index = self.index + 1
        else:
            self.index = 0
            self.buffer[index] = experience

        return self.index


    def random_sample(self, batch_size)->tuple:
        '''
        把buffer里的数据随机抽样出batch_size个，然后进行解析。\n

        Examples:
            tup1 = ([1,2,3], 2, 3, 4)\n
            tup2 = ([2,3,4], 2, 3, 4)\n
            tup3 = ([3,4,5], 2, 3, 4)\n
            tup4 = ([4,5,6], 2, 3, 4)\n
            全部push进buffer\n
            如果batch_size=2，那么输出的states例子为 \n
            tensor([[1, 2, 3], [3, 4, 5]]); states[0] = tensor([1, 2, 3])

        :param batch_size: 随机抽取的样本的数量
        :return: 对抽取的所有样本的解析
                 states: tensor; actions: int; rewards: float; states_future: tensor
        '''
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, states_future = zip(*batch)
        states = tensor(states)
        return states, actions, rewards, states_future
