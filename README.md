# Tensorflow for poets

* you will learn
    1 安装和运行TensorFlow Docker镜像
    2 使用python 训练图像分类器
    3 使用训练的分类器对图像进行分类

## 安装

安装tensorflow docker镜像
```
[zf@localhost ~]$ docker run -it tensorflow/tensorflow:latest-devel
root@xxxxxxx:~# 
```

测试是否正常安装
```
root@xxxxxxx:~# python
Python 2.7.12 (default, Nov 19 2016, 06:48:10) 
[GCC 5.4.0 20160609] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
Hello, TensorFlow!
```
安装正常，则输出“Hello, TensorFlow!”

## 数据集和训练程序

退出上述镜像实例，进入`/home`下载数据集
```
# ctrl-D if you're still in Docker and then:
cd $HOME
mkdir image_classification_tf_docker
cd image_classification_tf_docker
curl -O http://download.tensorflow.org/example_images/flower_photos.tgz
tar xzf flower_photos.tgz

```

解压后发现包含五个类别的图像
``daisy  dandelion  roses  sunflowers  tulips``

为加速训练可以暂时只训练二分类的分类器，删除其他三种类别的数据
```
# At your normal prompt, not inside Docker
cd $HOME/image_classification_tf_docker/flower_photos
rm -rf dandelion sunflowers tulips
```

将训练数据与docker镜像进行关联，以便进行后续训练
```
docker run -it -v $HOME/image_classification_tf_docker:/image_classification_tf_docker tensorflow/tensorflow:latest-devel
```

此时处于docker的context之中，`/`则是docker镜像的根目录，可以看到tensorflow和image_classification_tf_docker的语境
```
root@xxxxxx:/# ls 
bazel  dev   lib    mnt   root            sbin  tensorflow  usr
bin    etc   lib64  opt   run             srv   image_classification_tf_docker    var
boot   home  media  proc  run_jupyter.sh  sys   tmp
```

使用git获取最新的tensorflow代码
```
cd /tensorflow
git pull
```

示例代码位于`/tensorflow/tensorflow/examples/image_retraining/`之中。

## retrain Inception v3网络

```
# In Docker
python tensorflow/examples/image_retraining/retrain.py \
--bottleneck_dir=/image_classification_tf_docker/bottlenecks \
--how_many_training_steps 500 \
--model_dir=/image_classification_tf_docker/inception \
--output_graph=/image_classification_tf_docker/retrained_graph.pb \
--output_labels=/image_classification_tf_docker/retrained_labels.txt \
--image_dir /image_classification_tf_docker/flower_photos
```

该脚本将下载预训练的Inception v3模型，移除老模型的最后一层，并使用刚才下载的flower数据进行训练（二分类，最后一层两个神经元）。训练迭代500次之后结束：
```
2017-04-12 03:01:04.902830: Step 480: Train accuracy = 99.0%
2017-04-12 03:01:04.902895: Step 480: Cross entropy = 0.029856
2017-04-12 03:01:04.963329: Step 480: Validation accuracy = 98.0% (N=100)
2017-04-12 03:01:05.606954: Step 490: Train accuracy = 99.0%
2017-04-12 03:01:05.607016: Step 490: Cross entropy = 0.042138
2017-04-12 03:01:05.664779: Step 490: Validation accuracy = 99.0% (N=100)
2017-04-12 03:01:06.277076: Step 499: Train accuracy = 100.0%
2017-04-12 03:01:06.277143: Step 499: Cross entropy = 0.032325
2017-04-12 03:01:06.349133: Step 499: Validation accuracy = 98.0% (N=100)
Final test accuracy = 97.2% (N=145)
Converted 2 variables to const ops.
```

与其他DL框架的输出类似，
Train accuracy是训练过程中，在当前batch中图像分类准确率
Validation accuracy是在验证集batch上的图像分类准确率

## 使用训练好的模型

常用python或者C++接口部署训练好的模型。

编辑如下所示的label_image.py
```
import tensorflow as tf, sys

# change this as you see fit
image_path = sys.argv[1]

# Read in the image_data
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("/image_classification_tf_docker/retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("/image_classification_tf_docker/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})
    
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))
```

放入docker镜像的context之中，即`$HOME/image_classification_tf_docker`中。重启docker镜像

```
docker run -it -v $HOME/image_classification_tf_docker:/image_classification_tf_docker  tensorflow/tensorflow:latest-devel 
```

使用label_image.py对未知类别的图像进行分类。
```
root@xxxxxx:~# python /image_classification_tf_docker/label_image.py /image_classification_tf_docker/flower_photos/daisy/21652746_cc379e0eea_m.jpg
daisy (score = 0.99976)
roses (score = 0.00024)

root@6dbc56369bb4:~# python /image_classification_tf_docker/label_image.py /image_classification_tf_docker/flower_photos/roses/2414954629_3708a1a04d.jpg
roses (score = 0.99546)
daisy (score = 0.00454)
```

预测结果显示，训练好的分类器能够准确地预测出图像的类别。

## 其他超参数

试验中也可对`--learning_rate` 和`--train_batch_size`进行调整，同caffe等框架类似，学习率是应用于每一个batch的，当batchsize较大时，learning_rate应当降低。

## 定制自己的图像分类器

上述过程实现了rose和daisy两个类别的分类器，遵循同样的步骤，可以实现其他的图像分类器。可以使用[fatkun](https://chrome.google.com/webstore/detail/fatkun-batch-download-ima/nnjjahlikiabnchcpehcpkdeckfgnohf?hl=zh-CN)批量下载图片的工具组建数据集。

* 使用google图片搜索一个类别

![horse][horse]

* 使用fatkun进行批量下载

![down][down]

[horse]:imgs/horse.png
[down]:imgs/download.png

详细可[参考这里](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/index.html#0)