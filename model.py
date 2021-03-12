# Multi-structure Regions of Interest
# 
# References : 
#       CNN structure based on VGG16, https://github.com/ry/tensorflow-vgg16/blob/master/vgg16.py
#       Channel independent feature maps (3D features) using https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#depthwise_conv2d_native 
#       GAP based on https://github.com/jazzsaxmafia/Weakly_detector/blob/master/src/detector.py
#       Conv2d layer based on https://github.com/carpedm20/DCGAN-tensorflow/blob/master/ops.py

#import tensorflow as tf
import keras
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import resnet_499
import numpy as np
import _pickle as cPickle
from params import CNNParams, HyperParams

hyper     = HyperParams(verbose=False)
cnn_param = CNNParams(verbose=False)

def print_model_params(verbose=True):
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        if verbose: print("name: " + str(variable.name) + " - shape:" + str(shape))
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        if verbose: print("variable parameters: " , variable_parametes)
        total_parameters += variable_parametes
    if verbose: print("total params: ", total_parameters)
    return total_parameters

class CNN():
    
       
    def image_conversion_scaling(self, image):
        # Conversion to bgr and mean substraction is common with VGGNET
        # Because pre-trained values use them, https://arxiv.org/pdf/1409.1556.pdf
        image *= 255.
        r, g, b = tf.split(image, 3, 3)
        VGG_MEAN = [103.939, 116.779, 123.68]
        return tf.concat([b-VGG_MEAN[0], g-VGG_MEAN[1], r-VGG_MEAN[2]], 3)


    def build(self, image):
        model = resnet_499.ResNet152(include_top=False, weights='imagenet')#(?,7,7,2048)
        
        image = self.image_conversion_scaling(image)
        
        # this is a replcement of last FCL layer from VGG (common in GAP & GMP models)
        # this layer does not have non-nonlinearity
        conv_last = model(image)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(np.shape(conv_last))
        gap       = tf.reduce_mean(conv_last, [1,2])

        with tf.variable_scope("GAP"):
            gap_w = tf.get_variable("W", shape=cnn_param.layer_shapes['GAP/W'],
                    initializer=tf.random_normal_initializer(stddev=hyper.stddev))

        class_prob = tf.matmul(gap, gap_w)

        # print_model_params()
        return conv_last, gap, class_prob

    def p(self,t):
        print (t.name, t.get_shape())

    def get_classmap(self, class_, conv_last):
        with tf.variable_scope("GAP", reuse=True):
            class_w = tf.gather(tf.transpose(tf.get_variable("W")), class_)
            class_w = tf.reshape(class_w, [-1, cnn_param.last_features, 1]) 
        conv_last_ = tf.image.resize_bilinear(conv_last, [hyper.image_h, hyper.image_w])
        conv_last_ = tf.reshape(conv_last_, [-1, hyper.image_h*hyper.image_w, cnn_param.last_features]) 
        classmap   = tf.reshape(tf.matmul(conv_last_, class_w), [-1, hyper.image_h,hyper.image_w])
        return classmap

