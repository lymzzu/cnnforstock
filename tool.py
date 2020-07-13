# -*- coding: utf-8 -*-
"""
Created on Tue May 19 21:25:53 2020

@author: hunan
"""

import numpy as np
#import talib
import cv2
import tushare as ts
import matplotlib.pyplot as plt
import mpl_finance as mpf


def fig(data,path):
    fig = plt.figure(figsize=(24, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks(range(0, len(data['date']), 50))
    plt.xlim(0, len(data))
    plt.ylim(data['low'].min(),data['high'].max())
    #ax.plot(data['close'], label='10 日均线')
    #ax.plot(data['open'], label='10 日均线')
    mpf.candlestick2_ochl(ax, data['open'], data['close'], data['high'], data['low'],width=1, colorup='r', colordown='green',alpha=0.6)
    plt.savefig(path)
    #plt.close(fig)
    #img = cv2.imread("x.png")
    #return img
    #plt.show()
if __name__ == '__main__':
    data = ts.get_k_data('002320')
    img  = fig(data,'x')

