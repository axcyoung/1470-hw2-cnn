from __future__ import absolute_import

import os
import tensorflow as tf
import numpy as np
import random
import math

def conv2d(inputs, filters, strides, padding):
    """
    Performs 2D convolution given 4D inputs and filter Tensors.
    :param inputs: tensor with shape [num_examples, in_height, in_width, in_channels]
    :param filters: tensor with shape [filter_height, filter_width, in_channels, out_channels]
    :param strides: MUST BE [1, 1, 1, 1] - list of strides, with each stride corresponding to each dimension in input
    :param padding: either "SAME" or "VALID", capitalization matters
    :return: outputs, NumPy array or Tensor with shape [num_examples, output_height, output_width, output_channels]
    """
    input_shape = np.shape(inputs)
    num_examples = input_shape[0]
    in_height = input_shape[1]
    in_width = input_shape[2]
    input_in_channels = input_shape[3]

    filter_shape = np.shape(filters)
    filter_height = filter_shape[0]
    filter_width = filter_shape[1]
    filter_in_channels = filter_shape[2]
    filter_out_channels = filter_shape[3]

    num_examples_stride = strides[0]
    strideY = strides[1]
    strideX = strides[2]
    channels_stride = strides[3]

    if(input_in_channels != filter_in_channels):
        raise ValueError('Input and filter in channels are not equal')

    #Add padding
    if (padding == "SAME"):
        padY = math.floor((filter_height - 1)/2)
        padX = math.floor((filter_width - 1)/2)
        inputs = np.pad(inputs, ((0,), (padY,), (padX,), (0,)), mode='constant', constant_values=0)
    else:
        padY = 0.0
        padX = 0.0

    #Create empty output matrix
    output = np.zeros((num_examples, int((in_height + 2*padY - filter_height) / strideY + 1), int((in_width + 2*padX - filter_width) / strideX + 1), filter_out_channels), dtype=np.float32)

    #Calculate centers of centers of kernels, assume filter sizes are odd
    half_height = math.floor(filter_height/2)
    half_width = math.floor(filter_width/2)

    start_h = half_height
    end_h = in_height - half_height - 1
    start_w = half_width
    end_w = in_width - half_width - 1

    for example in range(0, num_examples):
        for j in range(start_h, end_h+strideY, strideY):
            for i in range(start_w, end_w+strideX, strideX):
                for out_c in range(0, filter_out_channels):
                    output[example, j-start_h, i-start_w, out_c] += tf.tensordot(filters[:, :, :, out_c], inputs[example, j-half_height:j+half_height+1, i-half_width:i+half_width+1, :], axes=([0,1,2],[0,1,2]))

    return output

def same_test_0():
    '''
    Simple test using SAME padding to check out differences between 
    own convolution function and TensorFlow's convolution function.

    NOTE: DO NOT EDIT
    '''
    imgs = np.array([[2,2,3,3,3],[0,1,3,0,3],[2,3,0,1,3],[3,3,2,1,2],[3,3,0,2,3]], dtype=np.float32)
    imgs = np.reshape(imgs, (1,5,5,1))
    filters = tf.Variable(tf.random.truncated_normal([2, 2, 1, 1],
                                dtype=tf.float32,
                                stddev=1e-1),
                                name="filters")
    my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="SAME")
    tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="SAME")
    print("SAME_TEST_0:", "my conv2d:", my_conv[0][0][0], "tf conv2d:", tf_conv[0][0][0].numpy())

def valid_test_0():
    '''
    Simple test using VALID padding to check out differences between 
    own convolution function and TensorFlow's convolution function.

    NOTE: DO NOT EDIT
    '''
    imgs = np.array([[2,2,3,3,3],[0,1,3,0,3],[2,3,0,1,3],[3,3,2,1,2],[3,3,0,2,3]], dtype=np.float32)
    imgs = np.reshape(imgs, (1,5,5,1))
    filters = tf.Variable(tf.random.truncated_normal([2, 2, 1, 1],
                                dtype=tf.float32,
                                stddev=1e-1),
                                name="filters")
    my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
    tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
    print("VALID_TEST_0:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def valid_test_1():
    '''
    Simple test using VALID padding to check out differences between 
    own convolution function and TensorFlow's convolution function.

    NOTE: DO NOT EDIT
    '''
    imgs = np.array([[3,5,3,3],[5,1,4,5],[2,5,0,1],[3,3,2,1]], dtype=np.float32)
    imgs = np.reshape(imgs, (1,4,4,1))
    filters = tf.Variable(tf.random.truncated_normal([3, 3, 1, 1],
                                dtype=tf.float32,
                                stddev=1e-1),
                                name="filters")
    my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
    tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
    print("VALID_TEST_1:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def valid_test_2():
    '''
    Simple test using VALID padding to check out differences between 
    own convolution function and TensorFlow's convolution function.

    NOTE: DO NOT EDIT
    '''
    imgs = np.array([[1,3,2,1],[1,3,3,1],[2,1,1,3],[3,2,3,3]], dtype=np.float32)
    imgs = np.reshape(imgs, (1,4,4,1))
    filters = np.array([[1,2,3],[0,1,0],[2,1,2]]).reshape((3,3,1,1)).astype(np.float32)
    my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
    tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
    print("VALID_TEST_1:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def main():
    # TODO: Add in any tests you may want to use to view the differences between your and TensorFlow's output
    valid_test_1()
    valid_test_2()
    return

if __name__ == '__main__':
    main()