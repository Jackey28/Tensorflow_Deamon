很多的深度学习框架都有以MNIST为数据集的demo，MNIST是很好的手写数字数据集。在网上很容易找到资源，但是下载下来的文件并不是普通的图片格式。不转换为图片格式也可以用。但有时，我们希望得到可视化的图片格式。

MNIST数据集包含4个文件：

train-images-idx3-ubyte:training set images 
train-labels-idx1-ubyte:training set labels 
t10k-images-idx3-ubyte: test set images 
t10k-labels-idx1-ubyte: test set labels

文件的格式很简单，可以理解为一个很长的一维数组。

测试图像(rain-images-idx3-ubyte)与训练图像(train-images-idx3-ubyte)由5部分组成：

32bits int

(magic number)
	

32bits int

图像个数
	

32bits int

图像高度28
	

32bits int

图像宽度28
	

像素值

（pixels）

 

测试标签(t10k-labels-idx1-ubyte)与训练标签(train-labels-idx1-ubyte)由3部分组成：

32bits int

(magic number)
	

32bits int

图像个数
	

标签

（labels）
