
@[TOC](文章目录)

---

# 前言
**使用 LSTM 对销售额预测（Python代码）**
大家经常会遇到一些需要预测的场景，比如预测品牌销售额，预测产品销量。
今天给大家分享一波使用 LSTM 进行端到端时间序列预测的完整代码和详细解释。
我们先来了解两个问题：

 - 什么是时间序列分析？
 - 什么是 LSTM？
---

`提示：以下是本篇文章正文内容，下面案例可供参考`

# 一、什么是时间序列分析？
时间序列表示基于时间顺序的一系列数据。它可以是秒、分钟、小时、天、周、月、年。未来的数据将取决于它以前的值。
在现实世界的案例中，我们主要有两种类型的时间序列分析：

- 单变量时间序列
- 多元时间序列
对于单变量时间序列数据，我们将使用单列进行预测。
![在这里插入图片描述](https://img-blog.csdnimg.cn/66c29651f69e4a6b972414aee9b92673.png#pic_center)

正如我们所见，只有一列，因此即将到来的未来值将仅取决于它之前的值。
但是在多元时间序列数据的情况下，将有不同类型的特征值并且目标数据将依赖于这些特征。
![在这里插入图片描述](https://img-blog.csdnimg.cn/6f968faa151c491d8b5b99269fa90838.png#pic_center)

正如在图片中看到的，在多元变量中将有多个列来对目标值进行预测。（上图中“count”为目标值）

在上面的数据中，count不仅取决于它以前的值，还取决于其他特征。因此，要预测即将到来的count值，我们必须考虑包括目标列在内的所有列来对目标值进行预测。

在执行多元时间序列分析时必须记住一件事，我们需要使用多个特征预测当前的目标，让我们通过一个例子来理解：

在训练时，如果我们使用 5 列 [feature1, feature2, feature3, feature4, target] 来训练模型，我们需要为即将到来的预测日提供 4 列 [feature1, feature2, feature3, feature4]。



# 二、LSTM又是什么捏？
本文中不打算详细讨论LSTM。所以只提供一些简单的描述，如果你对LSTM没有太多的了解，可以参考我们以前发布的文章。

LSTM基本上是一个循环神经网络，能够处理长期依赖关系。

假设你在看一部电影。所以当电影中发生任何情况时，你都已经知道之前发生了什么，并且可以理解因为过去发生的事情所以才会有新的情况发生。RNN也是以同样的方式工作，它们记住过去的信息并使用它来处理当前的输入。RNN的问题是，由于渐变消失，它们不能记住长期依赖关系。因此为了避免长期依赖问题设计了lstm。

现在我们讨论了时间序列预测和LSTM理论部分。让我们开始编码。

让我们首先导入进行预测所需的库：
## 1.引入库
代码如下（示例）：

```python

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
```

## 2.加载数据，并检查输出
代码如下（示例）：

```python
df=pd.read_csv("train.csv",parse_dates=["Date"],index_col=[0])
df.head()
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/48900856fdea4ab88b0d247c3c938d07.png#pic_center)


```python
df.tail()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/9b68128d2b294467839f913b97264da8.png#pic_center)
现在让我们花点时间看看数据：csv文件中包含了谷歌从2001-01-25到2021-09-29的股票数据，数据是按照天数频率的。

[如果您愿意，您可以将频率转换为“B”[工作日]或“D”，因为我们不会使用日期，我只是保持它的现状。]

这里我们试图预测“Open”列的未来值，因此“Open”是这里的目标列。
让我们看一下数据的形状：

```python
df.shape
(5203,5)
```

现在让我们进行训练测试拆分。这里我们不能打乱数据，因为在时间序列中必须是顺序的。

```python
test_split=round(len(df)*0.20)
df_for_training=df[:-1041]
df_for_testing=df[-1041:]
print(df_for_training.shape)
print(df_for_testing.shape)

(4162, 5)
(1041, 5)
```
可以注意到数据范围非常大，并且它们没有在相同的范围内缩放，因此为了避免预测错误，让我们先使用MinMaxScaler缩放数据。(也可以使用StandardScaler)

```python
scaler = MinMaxScaler(feature_range=(0,1))
df_for_training_scaled = scaler.fit_transform(df_for_training)
df_for_testing_scaled=scaler.transform(df_for_testing)
df_for_training_scaled
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/1a304807930149b0b3b0d4fbcde7d9c1.png#pic_center)
将数据拆分为X和Y，这是最重要的部分，正确阅读每一个步骤。

```python

def createXY(dataset,n_past):
  dataX = []
  dataY = []
  for i in range(n_past, len(dataset)):
          dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
          dataY.append(dataset[i,0])
  return np.array(dataX),np.array(dataY)

trainX,trainY=createXY(df_for_training_scaled,30)
testX,testY=createXY(df_for_testing_scaled,30)
```

让我们看看上面的代码中做了什么：

N_past是我们在预测下一个目标值时将在过去查看的步骤数。

这里使用30，意味着将使用过去的30个值(包括目标列在内的所有特性)来预测第31个目标值。

因此，在trainX中我们会有所有的特征值，而在trainY中我们只有目标值。

让我们分解for循环的每一部分：

对于训练，dataset = df_for_training_scaled, n_past=30
当i= 30：

```python
data_X.addend (df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
```
从n_past开始的范围是30，所以第一次数据范围将是-[30 - 30,30,0:5] 相当于 [0:30,0:5]

因此在dataX列表中，df_for_training_scaled[0:30,0:5]数组将第一次出现。

现在, dataY.append(df_for_training_scaled[i,0])

i = 30，所以它将只取第30行开始的open(因为在预测中，我们只需要open列，所以列范围仅为0，表示open列)。

第一次在dataY列表中存储df_for_training_scaled[30,0]值。

所以包含5列的前30行存储在dataX中，只有open列的第31行存储在dataY中。然后我们将dataX和dataY列表转换为数组，它们以数组格式在LSTM中进行训练。

我们来看看形状。

```python

print("trainX Shape-- ",trainX.shape)
print("trainY Shape-- ",trainY.shape)

(4132, 30, 5)
(4132,)

print("testX Shape-- ",testX.shape)
print("testY Shape-- ",testY.shape)

(1011, 30, 5)
(1011,)
```
4132 是 trainX 中可用的数组总数，每个数组共有 30 行和 5 列， 在每个数组的 trainY 中，我们都有下一个目标值来训练模型。

让我们看一下包含来自 trainX 的 (30,5) 数据的数组之一 和 trainX 数组的 trainY 值：

```python
print("trainX[0]-- \n",trainX[0])
print("trainY[0]-- ",trainY[0])
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/a97c854058274017af9453057a74d738.png#pic_center)
如果查看 trainX[1] 值，会发现到它与 trainX[0] 中的数据相同（第一列除外），因为我们将看到前 30 个来预测第 31 列，在第一次预测之后它会自动移动 到第 2 列并取下一个 30 值来预测下一个目标值。

让我们用一种简单的格式来解释这一切：

```python
trainX — — →trainY

[0 : 30,0:5] → [30,0]

[1:31, 0:5] → [31,0]

[2:32,0:5] →[32,0]
```
像这样，每个数据都将保存在 trainX 和 trainY 中。
## 3.模型建立

现在让我们训练模型，我使用 girdsearchCV 进行一些超参数调整以找到基础模型。

```python
def build_model(optimizer):
  grid_model = Sequential()
  grid_model.add(LSTM(50,return_sequences=True,input_shape=(30,5)))
  grid_model.add(LSTM(50))
  grid_model.add(Dropout(0.2))
  grid_model.add(Dense(1))

grid_model.compile(loss = 'mse',optimizer = optimizer)
  return grid_modelgrid_model = KerasRegressor(build_fn=build_model,verbose=1,validation_data=(testX,testY))

parameters = {'batch_size' : [16,20],
            'epochs' : [8,10],
            'optimizer' : ['adam','Adadelta'] }

grid_search = GridSearchCV(estimator = grid_model,
                          param_grid = parameters,
                          cv = 2)
```

如果你想为你的模型做更多的超参数调整，也可以添加更多的层。但是如果数据集非常大建议增加 LSTM 模型中的时期和单位。

在第一个 LSTM 层中看到输入形状为 (30,5)。它来自 trainX 形状。

```python
(trainX.shape[1],trainX.shape[2]) → (30,5)
```
现在让我们将模型拟合到 trainX 和 trainY 数据中。

```python
grid_search = grid_search.fit(trainX,trainY)
```
由于进行了超参数搜索，所以这将需要一些时间来运行。

你可以看到损失会像这样减少：

![在这里插入图片描述](https://img-blog.csdnimg.cn/a0ce591288234a21b74079ede41f88b4.png#pic_center)
现在让我们检查模型的最佳参数。

```python
grid_search.best_params_

{‘batch_size’: 20, ‘epochs’: 10, ‘optimizer’: ‘adam’}
```
将最佳模型保存在 my_model 变量中。

```python
my_model=grid_search.best_estimator_.model
```
## 4.模型检验
现在可以用测试数据集测试模型。

```python
prediction=my_model.predict(testX)
print("prediction\n", prediction)
print("\nPrediction Shape-",prediction.shape)
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/7d47c850b42946a993c9dc44703803f1.png#pic_center)
最后绘制一个图来对比我们的 pred 和原始数据。

```python
plt.plot(original, color = 'red', label = 'Real Stock Price')
plt.plot(pred, color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/c9f8162f15f74ed5b4ba79b53524630a.png#pic_center)



---

# 总结
木有总结 嘻嘻嘻
