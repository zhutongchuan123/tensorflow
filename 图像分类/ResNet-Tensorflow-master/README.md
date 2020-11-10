# ResNet in TensorFlow

Resnet相关资料
Implemenation of [Deep Residual Learning for Image
Recognition](http://arxiv.org/abs/1512.03385). 

MIT license. Contributions welcome.

## 基于源代码做的一些修改
1. 修改比较旧的，已经被抛弃的tensorflow的调用方法改成我使用的版本的，比如将tf.op_scope 改成 tf.name_scope。
2. 训练自己的数据的脚本。原代码只给了cifar和imagenet两个数据集的处理方法，我加了一个脚本可以处理任何形式的训练数据集。
3. 添加了预测的代码。
4. 修改了原来的代码使其能够在预训练的模型上继续训练。

## 具体用法
1. 准备自己的数据集
需要准备一个文本文件，里面包含了你的训练集的图片的绝对路径，以及对应的标签信息，标签必须是整数，每个标签对应一个，用tab分隔开。
然后将这个文本文件的路径作为参数，运行train_yourown.py 脚本。其他的设置比如具体用多少层的resnet，learning-rate之类就跟原代码一样设置就可以了。
如何传参详情参考tf.app.flags的用法。
2. 用训练好的模型预测
运行guess.py 的代码，你只需要修改下面几个参数：
data_dir :要测试的图片文件夹所在的路径
model_dir:训练好的模型所在的文件夹路径
ckpt_file:具体用哪个模型
target:预测输出的结果，总共包含三列，图片的路径，预测的标签，和预测的score
label_list:你的模型的标签信息，顺序是整数从小到大对应的信息，比如一开始是男0女1，那么这里就是["Male","Female"].
3. 预训练的模型可以在原代码的data文件夹中找到bt文件，下载下来就可以用了，我的代码使用50层的resnet的，如果需要用其他层数的，需要再做相应修改，只要改几个参数就可以了。

## 待完善部分
1. 训练模型验证部分需要修改。
2. 修改传参方法，更方便使用。

