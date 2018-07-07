#Google机器学习公开课学习#1

###框架处理:
####监督式机器学习：
>
> Labels：标签是我们要预测的真实实物
>
> Features：特征，描述数据的输入变量
>
> Example：样本，对于新数据做出预测，有标签或者无标签，无标签就要开始分类。
>
>
> model：模型，将样本映射到预测标签
####框架处理习题：
>
> 1、无标签样本就是没有被标签的样本，比如一堆邮件不知道是辣鸡还是不是辣鸡
>
>
> 2、使用特征必须得可量化，喜欢这种不可以被量化



###深入了解机器学习
####从数据中学习：
>
> 通过线性回归，将所有的散点变成一个线，这就是我们的预测线。其中y = wx + b，比如不可能所有点都在线上，所以有b，bias就是偏差。
>
> 这个方法通过**方差**来实现
>$$
> L_2Loss = \sum_{(x,y)\in D} (y - prediction(x))^2
>$$
> 对所有样本进行求和，将观察值减去预测值平方和，就是一个线性回归模型。
>
> 然后求一个平均方差也就是那就给这个L2除以N即可，也就是求平均值。



### 降低损失

#### 迭代方法

>确定怎么样修改模型方向
>
>**计算梯度**
>
>就是求误差函数的导数，高效的更新模型
>
>获得数据以后，可以计算这些数据的误差梯度，然后通过副梯度得到更新参数方向。
>
>那就是通过求导，然后找到一个误差函数的最低点，然后进行修改。
>
>修改的时候，考虑到一个学习速度，有可能太快跳过了最低点，或者太慢然后计算量过大。
>
>两种方法：
>
>> 随机梯度下降法：一次一个样本，随机抽取
>>
>> 最小梯度下降法：一次一个数据集，10到1000个样本左右
>
>若干术语：
>
>> 超参数：就是一个可以在计算过程中调整的一个参数，和参数相对



### TensorFlow使用基本步骤

使用数据集：

https://storage.googleapis.com/mledu-datasets/california_housing_train.csv

```
       longitude  latitude  housing_median_age  total_rooms  total_bedrooms  \
count    17000.0   17000.0             17000.0      17000.0         17000.0
mean      -119.6      35.6                28.6       2643.7           539.4
std          2.0       2.1                12.6       2179.9           421.5
min       -124.3      32.5                 1.0          2.0             1.0
25%       -121.8      33.9                18.0       1462.0           297.0
50%       -118.5      34.2                29.0       2127.0           434.0
75%       -118.0      37.7                37.0       3151.2           648.2
max       -114.3      42.0                52.0      37937.0          6445.0

       population  households  median_income  median_house_value
count     17000.0     17000.0        17000.0             17000.0
mean       1429.6       501.2            3.9            207300.9
std        1147.9       384.5            1.9            115983.8
min           3.0         1.0            0.5             14999.0
25%         790.0       282.0            2.6            119400.0
50%        1167.0       409.0            3.5            180400.0
75%        1721.0       605.2            4.8            265000.0
max       35682.0      6082.0           15.0            500001.0
```

这里通过pandas的series数据格式中的describe方法获得关于这一套数据的一些分析结果，包括平均值(mean)，标准差(std)，最大值最小值，然后还有25%~75%分位数的数据值，50%就是中位数。

在学习中，使用了reindex的方法对着17000行数据进行了重新的排序，不过不用管，这里describe的结果是一样的。

> 第一步：定义特征并配置特征列
>
> > **分类数据**：在表格中，住房数据集没有任何分类特征，不过可以有一些比如家具风格或者是广告词
> >
> > **数值数据**：一种数字数据，以及可以看为数字的数据，有时也可以作为分类数据

通过直接的提取方式，比如

```python
# Define the input feature: total_rooms.
my_feature = california_housing_dataframe[["total_rooms"]]

# Configure a numeric feature column for total_rooms.
feature_columns = [tf.feature_column.numeric_column("total_rooms")]
```

第一个方法是获得数据中total_rooms的这列的所有数据并给一个新的dataset，然后第二个是获得对应Dataset的一个属性：

> 第二步：定义目标
>
> > 接下来是确定目标，就是我要将给定的东西和我要目标预测的数据进行一个学习，首先先获得目标的Dataset：

```python
# Define the label.
targets = california_housing_dataframe["median_house_value"]
```

> 第三步：配置LinearRegressor
>
> > 由于我们是使用了线性回归模型来实现学习，同时使用的是小批量随机梯度下降法SGD训练模型，所以使用learning_rate参数来控制梯度步长，这里还会使用到梯度裁剪（使用clip_gradients_by_norm）防止梯度在训练的时候不会变得太大，从而越过了最低点。

```python
# Use gradient descent as the optimizer for training the model.
my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

# Configure the linear regression model with our feature columns and optimizer.
# Set a learning rate of 0.0000001 for Gradient Descent.
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer
)
```

> 第四步：定义输入函数
>
> > 在配置完源和目标以后，就要将这两个放入输入函数中进行预处理，然后提供给LinearRegression计算。记住这里使用的是最小批量随机梯度下降法，所以说一次使用的是一个数据，在这里就知道了batch_size等于的是1。然后通过分组，并无限制的复制，得到一长串全部都是单个数据集的一个大数据，然后通过shuffle，随机排列所有的内容，并从中抽取10000个到新的ds当中，然后通过iterator迭代器来返回所有的数值数据对应的目标值。

```python
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
  
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
    # Construct a dataset, and configure batching/repeating
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified
    if shuffle:
      ds = ds.shuffle(buffer_size=10000)
    
    # Return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels
```

> 第五步：训练模型
>
> > 终于现在可以开始训练了，由预处理函数处理好的10000个随机的数据将会输入到LinearRegressor里面，然后开始学习。在这里，源代码将my_input_fn封装在了lambda当中然后赋值给input_fn，这样才能将参数传递进去。然后steps=100代表我们会训练100步

```python
_ = linear_regressor.train(
    input_fn = lambda:my_input_fn(my_feature, targets),
    steps=100
)
```

> 第六步：评估模型
>
> > 基于训练做一次预测，然后看我们的模型在训练期间和数据的拟合情况。

```
# Create an input function for predictions.
# Note: Since we're making just one prediction for each example, we don't 
# need to repeat or shuffle the data here.
prediction_input_fn =lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)

# Call predict() on the linear_regressor to make predictions.
predictions = linear_regressor.predict(input_fn=prediction_input_fn)

# Format predictions as a NumPy array, so we can calculate error metrics.
predictions = np.array([item['predictions'][0] for item in predictions])

# Print Mean Squared Error and Root Mean Squared Error.
mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)
```

经过运行，结果如下：

```concole
Mean Squared Error (on training data): 56367.025
Root Mean Squared Error (on training data): 237.417
```

注意，由于MSE是正常值的平方，所以不好看出结果，在这里使用标准差也就是RMSE来和原数据进行一个判断，取数据最大最小值的中值来判断。标准差是反映一组数据离散程度最常用的一种量化形式，是表示精确度的重要指标。说起标准差首先得搞清楚它出现的目的。我们使用方法去检测它，但检测方法总是有误差的，所以检测值并不是其真实值。检测值与真实值之间的差距就是评价检测方法最有决定性的指标。但是真实值是多少，不得而知。因此怎样量化检测方法的准确性就成了难题。这也是临床工作质控的目的：保证每批实验结果的准确可靠。

虽然样本的真实值是不可能知道的，但是每个样本总是会有一个真实值的，不管它究竟是多少。可以想象，一个好的检测方法，其检测值应该很紧密的分散在真实值周围。如果不紧密，与真实值的距离就会大，准确性当然也就不好了，不可能想象离散度大的方法，会测出准确的结果。因此，离散度是评价方法的好坏的最重要也是最基本的指标。

在这里是通过极差的形式来评判的，也就是看极差和标准差之间的差距。

代码如下：

```python
min_house_value = california_housing_dataframe["median_house_value"].min()
max_house_value = california_housing_dataframe["median_house_value"].max()
min_max_difference = max_house_value - min_house_value

print("Min. Median House Value: %0.3f" % min_house_value)
print("Max. Median House Value: %0.3f" % max_house_value)
print("Difference between Min. and Max.: %0.3f" % min_max_difference)
print("Root Mean Squared Error: %0.3f" % root_mean_squared_error)
```

结果如下：

```
Min. Median House Value: 14.999
Max. Median House Value: 500.001
Difference between Min. and Max.: 485.002
Root Mean Squared Error: 237.417
```

由于极差是判断离散度最差的一个数值，所以说，当前的标准差快到这个最差的一半了，很不好，所以必须要进行修改来进一步的缩小误差。这里有一些基本的策略来降低模型误差。

首先，可以了解一下根据总体摘要统计信息，预测和目标的符合情况。

```python
calibration_data = pd.DataFrame()
calibration_data["predictions"] = pd.Series(predictions)
calibration_data["targets"] = pd.Series(targets)
calibration_data.describe()
```

从这些数据上来看，偏差还是很大的，那么将均匀分布的随机数据样本，绘制可辩的散点图。

```python
sample = california_housing_dataframe.sample(n=300)
```

然后根据模型的偏差想和特征权重绘制学到的线，并绘制散点图：

```python
# Get the min and max total_rooms values.
x_0 = sample["total_rooms"].min()
x_1 = sample["total_rooms"].max()

# Retrieve the final weight and bias generated during training.
weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

# Get the predicted median_house_values for the min and max total_rooms values.
y_0 = weight * x_0 + bias 
y_1 = weight * x_1 + bias

# Plot our regression line from (x_0, y_0) to (x_1, y_1).
plt.plot([x_0, x_1], [y_0, y_1], c='r')

# Label the graph axes.
plt.ylabel("median_house_value")
plt.xlabel("total_rooms")

# Plot a scatter plot from our data sample.
plt.scatter(sample["total_rooms"], sample["median_house_value"])

# Display graph.
plt.show()
```

很明显，这条线和实际的区别差距是巨大的，那么就需要开始调整参数，进行一定的修改，使得模型更加的准确（在这里就是要调整超参数）

> 调整模型超参数
>
> > 这时候就是开始调参数了，网站上提供了这样的一个函数`train_model`来帮助我们进行参数的修改。具体的代码如下：

```
def train_model(learning_rate, steps, batch_size, input_feature="total_rooms"):
  """Trains a linear regression model of one feature.
  
  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    input_feature: A `string` specifying a column from `california_housing_dataframe`
      to use as input feature.
  """
  
  periods = 10
  steps_per_period = steps / periods

  my_feature = input_feature
  my_feature_data = california_housing_dataframe[[my_feature]]
  my_label = "median_house_value"
  targets = california_housing_dataframe[my_label]

  # Create feature columns
  feature_columns = [tf.feature_column.numeric_column(my_feature)]
  
  # Create input functions
  training_input_fn = lambda:my_input_fn(my_feature_data, targets, batch_size=batch_size)
  prediction_input_fn = lambda: my_input_fn(my_feature_data, targets, num_epochs=1, shuffle=False)
  
  # Create a linear regressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.LinearRegressor(
      feature_columns=feature_columns,
      optimizer=my_optimizer
  )

  # Set up to plot the state of our model's line each period.
  plt.figure(figsize=(15, 6))
  plt.subplot(1, 2, 1)
  plt.title("Learned Line by Period")
  plt.ylabel(my_label)
  plt.xlabel(my_feature)
  sample = california_housing_dataframe.sample(n=300)
  plt.scatter(sample[my_feature], sample[my_label])
  colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  root_mean_squared_errors = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    predictions = linear_regressor.predict(input_fn=prediction_input_fn)
    predictions = np.array([item['predictions'][0] for item in predictions])
    
    # Compute loss.
    root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(predictions, targets))
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    root_mean_squared_errors.append(root_mean_squared_error)
    # Finally, track the weights and biases over time.
    # Apply some math to ensure that the data and line are plotted neatly.
    y_extents = np.array([0, sample[my_label].max()])
    
    weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
    bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

    x_extents = (y_extents - bias) / weight
    x_extents = np.maximum(np.minimum(x_extents,
                                      sample[my_feature].max()),
                           sample[my_feature].min())
    y_extents = weight * x_extents + bias
    plt.plot(x_extents, y_extents, color=colors[period]) 
  print("Model training finished.")

  # Output a graph of loss metrics over periods.
  plt.subplot(1, 2, 2)
  plt.ylabel('RMSE')
  plt.xlabel('Periods')
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(root_mean_squared_errors)

  # Output a table with calibration data.
  calibration_data = pd.DataFrame()
  calibration_data["predictions"] = pd.Series(predictions)
  calibration_data["targets"] = pd.Series(targets)
  display.display(calibration_data.describe())

  print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)
```

可以看出，这一段代码是对之前的一个整合，并封装到了一个train_model的函数当中，其中所有的参数可以通过函数的参数进行调整。具体调整如下：

```python
train_model(
    learning_rate=0.00001,
    steps=100,
    batch_size=1
)
```

## 任务 1：使 RMSE 不超过 180

调整模型超参数，以降低损失和更符合目标分布。
约 5 分钟后，如果您无法让 RMSE 低于 180，请查看解决方案，了解可能的组合。

在这个任务下要对参数进行修改，比如批处理组大小，训练的步数以及学习速度。在这里首先先放到比较大的步数进行计算：

```
train_model(
    learning_rate=0.00001,
    steps=1000,
    batch_size=1
)
```

不断的实时查看数据，然后知道一个比较好的结果出现了以后就可以记录当前的步数是多少了。

```concole
RMSE (on training data):
  period 00 : 225.63     The present period is:  0
  period 01 : 214.64     The present period is:  1
  period 02 : 204.24     The present period is:  2
  period 03 : 195.15     The present period is:  3
  period 04 : 187.70     The present period is:  4
  period 05 : 181.21     The present period is:  5
  period 06 : 176.21     The present period is:  6
  period 07 : 172.35     The present period is:  7
  period 08 : 170.01     The present period is:  8
  period 09 : 168.23     The present period is:  9
```

发现超过了180了任务完成。

## 任务 2：尝试其他特征

使用 `population` 特征替换 `total_rooms` 特征，看看能否取得更好的效果。

这部分不必超过 5 分钟。

代码如下：

```python
train_model(
    learning_rate=0.00001,
    steps=1000,
    batch_size=1,
    input_feature = 'population'
)
```

这时候发现，相同条件下，似乎效果变差了，所以population并不是一个很好的特征。



 # 合成特征和离群值

 ## 任务 1：尝试合成特征

`total_rooms` 和 `population` 特征都会统计指定街区的相关总计数据。

但是，如果一个街区比另一个街区的人口更密集，会怎么样？我们可以创建一个合成特征（即 `total_rooms` 与 `population` 的比例）来探索街区人口密度与房屋价值中位数之间的关系。

在以下单元格中，创建一个名为 `rooms_per_person` 的特征，并将其用作 `train_model()` 的 `input_feature`。

通过调整学习速率，您使用这一特征可以获得的最佳效果是什么？（效果越好，回归线与数据的拟合度就越高，最终 RMSE 也会越低。）

具体代码如下，dataframe可以直接进行计算，但是要保证数据的格式相同，否则会报错：

```python
california_housing_dataframe["rooms_per_person"] = (california_housing_dataframe["population"] / california_housing_dataframe["total_rooms"])

train_model(
    learning_rate=0.1,
    steps=800,
    batch_size=5,
    input_feature = "rooms_per_person"
)
```

这样计算结果RMSE下降到了118，比先前同样特征下的训练情况好很多，所以证明通过这个特征来训练效果会好很多！

 ## 任务 2：识别离群值

我们可以通过创建预测值与目标值的散点图来可视化模型效果。理想情况下，这些值将位于一条完全相关的对角线上。

使用您在任务 1 中训练过的人均房间数模型，并使用 Pyplot 的 `scatter()` 创建预测值与目标值的散点图。

您是否看到任何异常情况？通过查看 `rooms_per_person` 中值的分布情况，将这些异常情况追溯到源数据。

通过学习了pyplot，了解到了scatter散点图的画法以及subplot子图的一个布局方式，和原来的内容进行结合可以获得结果，可以看出，所有的点集中在了图像左侧且几乎平行于y轴，当然也有几个特异的点冒了出来。

```python
 X = calibration_data["predictions"]
 Y = calibration_data["targets"]
 plt.subplot(223)
 plt.scatter(X, Y)
```



 ## 任务 3：截取离群值

看看您能否通过将 `rooms_per_person` 的离群值设置为相对合理的最小值或最大值来进一步改进模型拟合情况。

以下是一个如何将函数应用于 Pandas `Series` 的简单示例，供您参考：

    clipped_feature = my_dataframe["my_feature_name"].apply(lambda x: max(x, 0))
上述 `clipped_feature` 没有小于 `0` 的值。

从图像上可以看出，冒出来的几个点大多数横坐标大于5，所以将当前的数值做进一步的调整，也就是将超出的值进行一个修改。

```python
california_housing_dataframe["rooms_per_person"] = (
    california_housing_dataframe["rooms_per_person"]).apply(lambda x: min(x, 5))

_ = california_housing_dataframe["rooms_per_person"].hist()
```

