# -*- coding:utf-8 -*-
# @Time : 2022/4/17 11:58
# @Author : aoweichen
# @File : timefeatures.py
# @Software: PyCharm

import warnings
from typing import List
import numpy as np
import pandas as pd
# pandas.tseries => pandas中处理时间序列的库
from pandas.tseries import offsets
#
from pandas.tseries.frequencies import to_offset

warnings.filterwarnings('ignore')

class TimeFeature:
    def __init__(self):
        pass
    # 该方法的功能类似于在类中重载 () 运算符，使得类实例对象可以像调用普通函数那样，以“对象名()”的形式使用。
    def __call__(self, index:pd.DatetimeIndex)->np.ndarray:
        pass
    # print()时
    # 默认情况下，__repr__() 会返回和调用者有关的 “类名+object at+内存地址”信息。当然，我们还可以通过在类中重写这个方法，从而实现当输出实例化对象时，输出我们想要的信息。
    def __repr__(self):
        return self.__class__.__name__+"()"

class SecondOfMinute(TimeFeature):
    """
        将每分钟中秒的编码为[-0.5,0.5]之间的数值
    """
    def __call__(self, index:pd.DatetimeIndex)->np.ndarray:
        return index.second/59.0-0.5

class MinuteOfHour(TimeFeature):
    """
        将每小时中的分钟编码为[-0.5,0.5]之间的数值
    """
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5

class HourOfDay(TimeFeature):
    """
        将每天中的小时编码为[-0.5,0.5]之间的数值
    """
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5

class DayOfWeek(TimeFeature):
    """将每周中的天编码为[-0.5,0.5]之间的数值"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5

class DayOfMonth(TimeFeature):
    """将每月中的天编码为[-0.5,0.5]之间的数值"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5

class DayOfYear(TimeFeature):
    """将每年天中的天编码为[-0.5,0.5]之间的数值"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5

class MonthOfYear(TimeFeature):
    """将每年中的月编码为[-0.5,0.5]之间的数值"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5

class WeekOfYear(TimeFeature):
    """将每年中的星期编码为[-0.5,0.5]之间的数值"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5

# 将时间戳转化为浮点数矩阵
def time_features_from_frequency_str(freq_str:str)->List[TimeFeature]:
    """
    依照传入的格式返回一个时间特征转换列表，eg:[MonthOfYear,...]
    返回适用于给定时间频率的时间特征列表

    :param freq_str: 形式为[多个][粒度]的频率串，如“12H”、“5min”、“1D”等。粒度指划分的精细程度，如天、日、小时等
    :return:
    """
    # 表示不同的时间粒度
    feature_by_offsets = {
        # offsets.YearEnd:时间戳表示以年结尾的话，就以对应value的方式编码 => []
        offsets.YearEnd:[],
        # offsets.QuarterEnd:时间戳表示以季度结尾的话，就以对应value的方式编码 => [MonthOfYear]
        offsets.QuarterEnd:[MonthOfYear],
        # offsets.MonthEnd:时间戳表示以月结尾的话，就以对应value的方式编码 => [MonthOfYear]
        offsets.MonthEnd:[MonthOfYear],
        # offsets.MonthEnd:时间戳表示以星期结尾的话，就以对应value的方式编码 => [DayOfMonth,WeekOfYear]
        offsets.Week:[DayOfMonth,WeekOfYear],
        # 依上类推
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        # 依上类推
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        # 依上类推
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        # 依上类推
        offsets.Minute: [MinuteOfHour,HourOfDay,DayOfWeek,DayOfMonth,DayOfYear],
        # 依上类推
        offsets.Second: [SecondOfMinute,MinuteOfHour,HourOfDay,DayOfWeek,DayOfMonth,DayOfYear],
    }
    # 将输入的freq_str转化为对应的offsets
    # 如"h"转化为offsets.Hour
    offset = to_offset(freq_str)
    # items()=>Python 字典(Dictionary) items() 函数以列表返回可遍历的(键, 值) 元组数组。
    # 依照传入的格式返回一个时间特征转换列表，eg:[MonthOfYear,...]
    for offset_type,feature_classes in feature_by_offsets.items():
        if isinstance(offset,offset_type):
            return [cls() for cls in feature_classes]

    # 当上面那个循环不成立的时候，报个错，提示输入格式不对
    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)

# 对时间进行编码
def time_features(dates,timeenc=1,freq='h'):
    """
        参数:
            timeenc:表示时间编码的格式
            ->timeenc == 1:数值矩阵[-0.5,0.5]
            ->timeenc == 0:时间矩阵[7,4,5,6]


            > `time_features` takes in a `dates` dataframe with a 'dates' column and extracts the date down to `freq` where freq can be any of the following if `timeenc` is 0:
            > * m - [month]
            > * w - [month]
            > * d - [month, day, weekday]
            > * b - [month, day, weekday]
            > * h - [month, day, weekday, hour]
            > * t - [month, day, weekday, hour, *minute]
            >
            > If `timeenc` is 1, a similar, but different list of `freq` values are supported (all encoded between [-0.5 and 0.5]):
            > * Q - [month]
            > * M - [month]
            > * W - [Day of month, week of year]
            > * D - [Day of week, day of month, day of year]
            > * B - [Day of week, day of month, day of year]
            > * H - [Hour of day, day of week, day of month, day of year]
            > * T - [Minute of hour*, hour of day, day of week, day of month, day of year]
            > * S - [Second of minute, minute of hour, hour of day, day of week, day of month, day of year]

            *minute returns a number from 0-3 corresponding to the 15 minute period it falls into.
    """
    if timeenc==0:
        # dataframe.apply(function,axis)对一行或一列做出一些操作（axis=1则为对某一列进行操作，此时，apply函数每次将dataframe的一行传给function，然后获取返回值，将返回值放入一个series）
        # 对dates的每一行进行操作，得到月份数据
        dates['month'] = dates.date.apply(lambda row:row.month,1)
        # 依上类推
        dates['day'] = dates.date.apply(lambda row:row.day,1)
        # 依上类推
        dates['weekday'] = dates.date.apply(lambda row:row.weekday(),1)
        # 依上类推
        dates['hour'] = dates.date.apply(lambda row:row.hour,1)
        # 依上类推
        dates['minute'] = dates.date.apply(lambda row:row.minute,1)
        # 依上类推
        dates['minute'] = dates.minute.map(lambda x:x//15)
        # freq对应的格式
        freq_map = {
            'y':[],'m':['month'],'w':['month'],'d':['month','day','weekday'],
            'b':['month','day','weekday'],'h':['month','day','weekday','hour'],
            't':['month','day','weekday','hour','minute'],
        }
        # 返回的对应的编码
        return dates[freq_map[freq.lower()]].values
    if timeenc==1:
        # 将时间变为DatetimeIndex格式
        dates = pd.to_datetime(dates.date.values)
        # np.vstack()=>
        # 返回另一种格式的数据=>区间在[-0.5,0.5]之间的数值矩阵
        return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)]).transpose([1,0])


def main():
    data = pd.read_csv("F:\AI\DL\TimeSeries\Informer\InformerCode\data\ETT\ETTh1.csv",encoding="GB18030")
    data['date'] = data['date']
    dates = data[['date']]
    dates['date'] = pd.to_datetime(dates.date)
    date = time_features(dates,1,'H')
    print(data['date'])
    print(date)


if __name__ == "__main__":
    main()