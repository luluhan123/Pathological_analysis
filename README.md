# 乳腺癌病理分割现有结果及后续安排

## 1. 现有项目结构

![20190823114140-pathological analysis.png](https://raw.githubusercontent.com/luluhan123/ImageforMarkdown/master/20190823114140-pathological%20analysis.png)

## 2. 现有细胞分割流程

```flowchart
start=>start: 开始
operation1=>operation: image split 成1024 * 1024
operation2=>operation: image transfer
operation3=>operation: image deconvolution 得到He
operation4=>operation: open morphological operation
operation5=>operation: close morphological operations
operation6=>operation: Otsu's thresholding
operation7=>operation: open morphological operations
operation8=>operation: remove small area and small holes
operation9=>operation: remove artifact
operation10=>operation: run adaptive multi-scale LOG filter
operation11=>operation: detect and segment nuclei using local maximum clustering
operation12=>operation: filter out small objects

start->operation1->operation2->operation3->operation4->operation5->operation6->operation7->operation8->operation9->operation10->operation11->operation12
```

### 说明：

1. image transfer一定要选取对比度鲜明的图像作为模板

2. 开闭操作一定选择椭圆核，这样可以得到更平滑的图像，而且得到的结果更倾向于椭圆，符合我们的要求

3. 移除小孔和识别伪影这里都是设置了面积的阈值，目前没有更好的方法

## 3. 现有问题及接下来需要进行的工作

![20190823153448-result5_5_3.png](https://raw.githubusercontent.com/luluhan123/ImageforMarkdown/master/20190823153448-result5_5_3.png)

1. 首先需要确定三个超参数：min_radius、max_radius、local_max_search_radius。我现在生成了三个超参数不同组合的图片，共有近千张，需要选出最优超参数。我建议找刘主任标注出来一张图像，然后进行比对。

2. 粗分割时可能会带有一些细胞质进来，导致分割结构不规则，在后面的细分割得到一些奇怪的结构

3. 对于伪影的处理，使用控制面积阈值的方法其实并不靠谱，希望可以有更优的方法

4. 得到细分割的结果之后，对每个细胞进行统计半径，面积，长宽比等特征，进行分类

5. 算法加速：建议使用Cython重构一下，我之前写过，出了点问题，就删掉了

6. 可以了解一下病理学知识，还有生物统计学

7. 可以写一个带有界面的勾画软件，把每次得到的结果发给刘主任，让他们使用软件进行修改，多次循环后得到的结果就可以用于深度学习训练了
