# FONet
Frequency Space Enhancement and Online Score Prediction for Temporally Adaptive RGBT Object Tracking

# 本文主要贡献如下：

1、为了令跟踪器对输入的模板和搜索图像有充分的信息交互，我们设计了一种结合频率注意力与空间增强模块的频率空间增强模块，旨在构建模板特征与搜索特征的通道与空间的联系，得到更加鲁棒的特征。

2、为了令跟踪器充分利用时间信息，我们设计了一种由两层多头注意力模块构成的OSP模块，通过将当前搜索嵌入向量与动态模板嵌入向量输入其中，引入时间信息，最后通过输出的置信度分数在线更新动态模板，有效适应目标外观随时间的变化。

3、大量实验表明，我们的方法在两个主流的RGBT跟踪数据集上实现了最先进的性能。
