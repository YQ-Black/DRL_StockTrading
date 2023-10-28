import pandas as pd
import os
import numpy as np

class CSVProcessor():
    '''
    一个类，用来对从YahooFinance上的数据做预处理。
    目前功能：
        1.data_pre_process：把csv文件转化成dataframe文件，并对其中的NA值做处理。返回值：数据的dataframe类型
    '''
    def data_pre_process(self, file_path):
        # 获取股票的公司名并输出
        file_name = os.path.basename(file_path)  # 获取文件名（包括后缀）
        file_name_without_ext = os.path.splitext(file_name)[0]
        print(f'这是{file_name_without_ext}的股票数据')

        # 把csv转换成dataframe
        df = pd.read_csv(file_path)

        # 对数据做初步处理                      inplace=True:得到的结果会覆盖原来的
        df.replace(0.0, np.nan, inplace=True)
        df.ffill()  # 向前插值法
        df.bfill() # 向后插值法
        df.dropna(inplace=True)             # give up data NA

        return df

# procecssor = CSVProcessor()
# procecssor.data_pre_process('D:\science\Design\数据\AAPL.csv')