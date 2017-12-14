# -*- coding:utf-8 -*-
import tensorflow as tf
import tensorboard
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot='True')


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def Weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    """从截断的正态分布中输出随机值。
生成的值服从具有指定平均值和标准偏差的正态分布，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择。

    """
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


"""
生成一个shape 里全为0.1的矩阵
"""


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


"""
x 表示输入的矩阵 ， W表示权值 strides 表示跨度步长[1,1,1,1]
规定第一个和第四个一定为 1 第二个1表示水平跨度为1 第三个1 表示垂直跨度唯1
padding
same 表示可以抽取到图片外面 超出部分用0填充 抽取生成的图片大小等于给定图片
valid 只在图片中取 抽取的生成的图片小于给定图片
"""


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')


"""
x表示输入图片
ksize 表示第1为第四位必须等于1 补偿为2 相当于把图片压缩
注意是polling 这个过程缩小图片
"""

#xs = tf.placeholder(tf.float32, [None, 784])  # 28 *28
with tf.name_scope('input'):
    xs = tf.placeholder(tf.float32, [None, 784],name='x_in')/255  # 28 *28
    ys = tf.placeholder(tf.float32, [None, 10],name='y_in')  #
    with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder(tf.float32,name='probabilityOfdopout')
x_image = tf.reshape(xs, [-1, 28, 28, 1])
"""
-1 表示把所有图片的维度先不管 sample 个数
1表示 channnel 这里图片是黑白的 所以为1
"""
print(x_image.shape)  # 想要输出图片维度 [n_samole,28,28,1]
with tf.name_scope('conv_1'):
    with tf.name_scope('W'):
       W_conv1 = Weight_variable([5, 5, 1, 32])  # 5*5的 patch  in size= 1 输入图片的厚度 out size = 32c 输出的高度是32
    with tf.name_scope('bias'):
        b_conv1 = bias_variable([32])
    with tf.name_scope('ReLU'):
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# relu非线性处理 output size = 28*28 *32 由于抽出的形式是same 形式so 图片的长度保持不变
with tf.name_scope('pool_1'):
    h_pool1 = max_pool_2x2(h_conv1)  # out put size =  14*14*32 说明这一层卷积层有32个filtive
"""
poll 的步伐 是二 so 图片被缩小为1/2
"""
#建立第二卷积层
with tf.name_scope('conv2'):
    with tf.name_scope('W'):
        W_conv2 = Weight_variable([5, 5, 32, 64])  # 5*5的 patch  in size= 1 输入图片的厚度 out size = 32c 输出的高度是32
    with tf.name_scope('bias'):
        b_conv2 = bias_variable([64])
    with tf.name_scope('ReLU'):
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # relu非线性处理 output size = 14*14*32
with tf.name_scope('pool_2'):
    h_pool2 = max_pool_2x2(h_conv2)  # out put size =  7*7*64
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])

# func1 layer
with tf.name_scope('func1'):
    with tf.name_scope('W'):
        W_fc1 = Weight_variable([7 * 7 * 64, 1024])  # 输入为7*7*64 输出是1024
    with tf.name_scope('bias'):
        b_fc1 = bias_variable([1024])
    with tf.name_scope('ReLU'):
       h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
    with tf.name_scope('drop'):
       h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)#注意 keep_prob 就用在这里

# func2 layer
with tf.name_scope('func1'):
    with tf.name_scope('W'):
        W_fc2 = Weight_variable([1024, 10])
    with tf.name_scope('bias'):
        b_fc2 = bias_variable([10])
with tf.name_scope('prediction'):
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#loss
with tf.name_scope('loss'):
    cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
with tf.name_scope('train'):
    train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess=tf.Session()
writer = tf.summary.FileWriter("/home/jackey/PycharmProjects/ML/CNN/logs/", sess.graph)
init = tf.global_variables_initializer()
sess.run(init)
for step in range(100):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    #   sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    #   sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob: 0.5})
    if step%5 ==0:
        #print compute_accuracy(mnist.test.images,mnist.test.labels)
        print(compute_accuracy(
            mnist.test.images[:1000], mnist.test.labels[:1000]))
#https://github.com/joopark28/deeplearning.git
"""

"""