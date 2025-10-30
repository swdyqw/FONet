# FONet
Frequency-Space Enhanced and Temporal Adaptative RGBT Object Tracking

# 本文主要贡献如下：

1、为了令跟踪器对输入的模板和搜索图像有充分的信息交互，我们设计了一种结合频率注意力与空间增强模块的频率空间增强模块，旨在构建模板特征与搜索特征的通道与空间的联系，得到更加鲁棒的特征。

2、为了令跟踪器充分利用时间信息，我们设计了一种由两层多头注意力模块构成的OSP模块，通过将当前搜索嵌入向量与动态模板嵌入向量输入其中，引入时间信息，最后通过输出的置信度分数在线更新动态模板，有效适应目标外观随时间的变化。

3、大量实验表明，我们的方法在两个主流的RGBT跟踪数据集上实现了最先进的性能。


## 发表期刊：Neurocomputing

## 论文地址：https://www.sciencedirect.com/science/article/pii/S0925231225009129

## 论文简介：

由于现有基于Transformer的RGBT跟踪方法对于模板与搜索图像交互不足，导致其在某些复杂的场景下，例如运动模糊、相机移动等场景，跟踪性能急剧下降。除此之外，由于目标跟踪中，对当前帧的预测结果通常与历史帧关联性比较强，然而目前大多数RGBT跟踪方法没有充分利用时间信息。

## 核心技术：

1、为了令跟踪器对输入的模板和搜索图像有充分的信息交互，我们设计了一种结合频率注意力与空间增强模块的频率空间增强模块，旨在构建模板特征与搜索特征的通道与空间的联系，得到更加鲁棒的特征。

2、为了令跟踪器充分利用时间信息，我们设计了一种由两层多头注意力模块构成的在线分数预测模块，通过将当前搜索嵌入向量与动态模板嵌入向量输入其中，引入时间信息，最后通过输出的置信度分数在线更新动态模板，有效适应目标外观随时间的变化。

3、大量实验表明，我们的方法在两个主流的RGBT跟踪数据集上实现了最先进的性能。

## abstract:

Recently, superior tracker of ViT in RGBT tracking has been widely applied due to its global and dynamic modeling capabilities. However, these ViT-based trackers often extract information from the template and search images independently, which leads to difficulties in focusing on the discriminative target features and makes them vulnerable to background interference. In addition, they typically neglect to associate the relationship between the current and historical frames, which causes drifting. To address these issues, we propose a RGB-T object tracking network based on frequency-space enhancement and temporal adaptation. We use pre-trained ConvMAE as the feature extraction backbone. Specifically, we design a frequency-space enhancement (FSE) module to establish associations between the template and search images in both the channel and spatial dimensions to extract more discriminative features. Besides, to adaptatively introduce temporal information, we propose an online score prediction (OSP) module to link the current frame with historical frames. OSP calculates prediction score based on the correlation between the current and historical frames, guiding the FETA to dynamically update the online template. Extensive quantitative and qualitative experiments on two RGBT benchmark datasets RGBT234 and LasHeR demonstrate that our method outperforms existing state-of-the-art trackers in terms of both accuracy and robustness.
