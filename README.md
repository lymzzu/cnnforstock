# cnnforstock
利用残差网络预测K线的涨跌
怎么说呢？这是一个失败的实验，实验中尝试了多种残差网络有10层，18层，34层。
各种超参数，训练了100个epochs，但是结果loss保持在0.6到0.7之间，因为是个二分类问题，
所以判断对的概率大约在0.5左右，即并没有学习到什么东西，但是还是要总结一下失败的经验。

利用股票的直接k线的想法是基于cnn的原理，cnn厉害的地方在于可以学习到局部特征之间的相关性，
如果直接利用特征的话，当特征列之间的顺序改变就会导致特征之间的相关性发生改变，从而影响识
别的效果。基于这个接入点本实验采取168支股票从2015-01-01到2019-01-01的股票日行情。然后用
前30天的k线图预测后五天的涨跌，若后五天中存在一天的收盘价涨幅超过3%打标签为1，反之为0。统
计训练集总共有121096个样本，其中正样本负样本几乎平衡。数据处理：图像首先resize到（64，64），
然后归一化到（0，1）最后利用mean=(0.5,0.5,0.5),std = (0.5,0.5,0.5)的标准化实验中考虑的残
差网络有10层，18层，34层其中卷积核（3，3）。batchsize=128, 100个epoch。
