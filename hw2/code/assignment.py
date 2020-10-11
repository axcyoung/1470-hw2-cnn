from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from convolution import conv2d

import os
import tensorflow as tf
import numpy as np
import random
import math

class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that 
        classifies images. Do not modify the constructor, as doing so 
        will break the autograder. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()

        self.batch_size = 64
        self.num_classes = 2
        self.loss_list = [] # Append losses to this list in training so you can visualize loss vs time in main
        self.learning_rate = .001
        self.dropout_rate = .3
        self.var_ep = .000001

        self.filter1 = tf.Variable(tf.random.truncated_normal([5,5,3,16], stddev=.1, dtype=tf.float32))
        self.filter2 = tf.Variable(tf.random.truncated_normal([5,5,16,20], stddev=.1, dtype=tf.float32))
        self.filter3 = tf.Variable(tf.random.truncated_normal([3,3,20,20], stddev=.1, dtype=tf.float32))
        self.W1 = tf.Variable(tf.random.truncated_normal([2*2*20, 2], stddev=.1, dtype=tf.float32))
        self.b1 = tf.Variable(tf.random.truncated_normal([2], stddev=.1, dtype=tf.float32))

    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of filter = (filter_height, filter_width, in_channels, out_channels)
        # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)

        layer1Output = tf.nn.conv2d(inputs, self.filter1, strides=[2,2], padding='SAME')
        meanVar1 = tf.nn.moments(layer1Output, axes=[0, 1, 2])
        layer1Output = tf.nn.batch_normalization(layer1Output, meanVar1[0], meanVar1[1], offset=None, scale=None, variance_epsilon=self.var_ep)
        layer1Output = tf.nn.relu(layer1Output)
        layer1Output = tf.nn.max_pool(layer1Output, [3,3], [2,2], padding='SAME')

        layer2Output = tf.nn.conv2d(layer1Output, self.filter2, strides=[1,1], padding='SAME')
        meanVar2 = tf.nn.moments(layer2Output, axes=[0, 1, 2])
        layer2Output = tf.nn.batch_normalization(layer2Output, meanVar2[0], meanVar2[1], offset=None, scale=None, variance_epsilon=self.var_ep)
        layer2Output = tf.nn.relu(layer2Output)
        layer2Output = tf.nn.max_pool(layer2Output, [2,2], [2,2], padding='SAME')
        
        if is_testing == True:
            layer3Output = conv2d(layer2Output, self.filter3, strides=[1,1,1,1], padding='SAME')
            layer3Output = tf.convert_to_tensor(layer3Output)
        else:
            layer3Output = tf.nn.conv2d(layer2Output, self.filter3, strides=[1,1,1,1], padding='SAME')
        meanVar3 = tf.nn.moments(layer3Output, axes=[0, 1, 2])
        layer3Output = tf.nn.batch_normalization(layer3Output, meanVar3[0], meanVar3[1], offset=None, scale=None, variance_epsilon=self.var_ep)
        layer3Output = tf.nn.relu(layer3Output)
        layer3Output = tf.nn.max_pool(layer3Output, 2, 2, padding='SAME')

        layer4Output = tf.nn.dropout(layer3Output, self.dropout_rate)
        layer4Output = tf.reshape(layer4Output, [layer4Output.shape[0], -1])
    
        logits = tf.matmul(layer4Output, self.W1) + self.b1

        return logits

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        Softmax is applied in this function.
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """ 
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT
        
        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs 
    and labels - ensure that they are shuffled in the same order using tf.gather.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: Optionally list of losses per batch to use for visualize_loss
    '''
    num_examples = train_labels.shape.as_list()[0]
    shuffle_indices = np.arange(0, num_examples)
    shuffle_indices = tf.random.shuffle(shuffle_indices)
    train_inputs = tf.gather(train_inputs, shuffle_indices)
    train_labels = tf.gather(train_labels, shuffle_indices)

    optimizer = tf.keras.optimizers.Adam(model.learning_rate)

    for i in range(0, num_examples, model.batch_size):
        input_batch = train_inputs[i:i + model.batch_size, :, :, :]
        input_batch = tf.image.random_flip_left_right(input_batch)
        
        label_batch = train_labels[i:i + model.batch_size]
        
        with tf.GradientTape() as tape:
            logits = model.call(input_batch)
            loss = model.loss(logits, label_batch)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        model.loss_list.append(loss)
        #print(model.accuracy(logits, label_batch))
        

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly 
    flip images or do any extra preprocessing.
    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """
    accuracies = []
    for i in range(0, test_labels.shape.as_list()[0], model.batch_size):
        input_batch = test_inputs[i:i + model.batch_size, :, :, :]
        
        label_batch = test_labels[i:i + model.batch_size]
        
        logits = model.call(input_batch, True)
        accuracies.append(model.accuracy(logits, label_batch))
    return np.mean(np.array(accuracies))
    

def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()  


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "dog"

    NOTE: DO NOT EDIT

    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    # Helper function to plot images into 10 columns
    def plotter(image_indices, label): 
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle("{} Examples\nPL = Predicted Label\nAL = Actual Label".format(label))
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i+1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(
                image_labels[ind], axis=0) == 0 else second_label
            ax.set(title="PL: {}\nAL: {}".format(pl, al))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
        
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images): 
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0): 
            correct.append(i)
        else: 
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()


def main():
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and 
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs. 
    
    CS1470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=70%.
    
    CS2470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=75%.
    
    :return: None
    '''
    training = get_data('data/train', 3, 5)
    train_inputs = training[0]
    train_labels = training[1]
    
    testing = get_data('data/test', 3, 5)
    test_inputs = testing[0]
    test_labels = testing[1]
    
    m = Model()
    for i in range(10):
        train(m, train_inputs, train_labels)
        
    print(test(m, test_inputs, test_labels))
    visualize_loss(m.loss_list)
    return


if __name__ == '__main__':
    main()
