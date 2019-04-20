import tensorflow as tf
import utils
import numpy as np
from PIL import Image,ImageOps
import resnet_model

class Model:
    def __init__(self, class_size, batch_size, new_height, new_width, learning_rate):
        self.batch_size = batch_size
        self.new_height = new_height
        self.new_width = new_width
        self.class_size = class_size
        self.layers = []
        self.learning_rate = 0.01

    def _init_placeholders(self):
        x = tf.placeholder(tf.float32, shape = [None,self.new_height,self.new_width,3], name = 'x')
        y = tf.placeholder(tf.float32, shape = [None,self.class_size], name = 'y')
        return x,y

    def _init_model(self,x):
        l1 =tf.layers.conv2d(x,32,[3,3],padding= "same",activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer(),name = 'conv1')
        l2 =tf.layers.max_pooling2d(l1,[2,2],strides = 2, padding= 'same', name = 'pool1')
        l3 =tf.layers.conv2d(l2,64,[3,3],padding ='same',strides = (2,2), activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer(),name = 'conv2')
        l4 =tf.layers.conv2d(l3,64, [3,3],padding = 'same',activation = tf.nn.relu,kernel_initializer = tf.contrib.layers.xavier_initializer(),name = 'conv3')
        l5 =tf.layers.conv2d(l4,64, [3,3],padding ='same',activation = tf.nn.relu,kernel_initializer = tf.contrib.layers.xavier_initializer(),name = 'conv4')
        l6 =tf.layers.max_pooling2d(l5,[2,2],strides = 2, padding= 'same', name = 'pool2')
        l7 =tf.layers.flatten(l6, name = 'flatten')
        l8 =tf.layers.dense(l7, 512, activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer(), name = 'FC1')
        logits = tf.layers.dense(l8,self.class_size, kernel_initializer = tf.contrib.layers.xavier_initializer(), name = 'FC2')
        return logits


    def _init_loss(self,y,logits):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y,logits = logits,name ='loss'))
        return loss

    def _init_optimizer(self,loss):
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        return optimizer


    def _build_model(self):
        x,y = self._init_placeholders()
        #logits = self._init_model(x)
        logits = resnet_model._init_resnet_model(x)
        loss = self._init_loss(y,logits)
        optimizer = self._init_optimizer(lr,loss)


    def __run__(self, epoch):
        self._build_model()
        train_data, train_ids, test_data, test_ids = utils.read('Images')
        perm = np.random.permutation(len(train_data))
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            idx = 0
            for i in range(epoch):
                mean_arr = np.array([1.214148920540898047e+02, 1.152152514116808391e+02, 9.971035506189132036e+01],dtype = np.uint8)
                cond = True
                while cond:
                    if idx + self.batch_size >= len(train_data):
                        index = perm[idx:]
                        cond = False
                    else:
                        index = perm[idx: idx + self.batch_size]

                    processed_imgs = self.resize_and_crop(train_data[index])
                    processed_test_imgs = self.resize_and_crop(test_data)
                    processed_imgs -= mean_arr
                    processed_test_imgs -= mean_arr
                    _,l = sess.run([optimizer,loss], feed_dict = {x:processed_imgs,y:train_ids[index]})
                    idx += self.batch_size
                    print("epoch: {0}, loss: {1} ".format(i,l))

            logits = sess.run(logits, feed_dict = {x:test_X, y:test_y_eye})
            preds = np.argmax(logits,axis =1)
            hits = test_label == preds
            acc = (np.count_nonzero(hits) / test_X.shape[0])*100
            print("accuracy : ", acc)
