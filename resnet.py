import tensorflow as tf
import numpy as np
import data

class resnet_model:
    def __init__(self,class_size,new_height,new_width,batch_size,lr):
        self.lr = lr
        self.class_size = class_size
        self.new_height = new_height
        self.new_width = new_width
        self.batch_size = batch_size
        #self.beta = tf.get_variable(name = 'beta', shape = [D,C], dtype = tf.float32, initializer = tf.zeros_initializer())
        #self.gamma = tf.get_variable(name = 'gamma', shape = [D,C], dtype = tf.float32, initializer = tf.ones_initializer())


    def _init_placeholders(self):
        x = tf.placeholder(tf.float32, shape = [None,self.new_height,self.new_width,3], name = 'x')
        y = tf.placeholder(tf.float32, shape = [None,self.class_size], name = 'y')
        return x,y
#conv1
    def _init_resnet_model(self,x):
        tf.reset_default_graph()
        l1 = tf.layers.conv2d(x,64,[7,7],strides = (2,2), padding = "same",activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer(),name = 'conv1')
        l2 = tf.nn.max_pool(l1, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
        l3 = self._res_block(l2,64)
        l4 = self._res_block(l3,64)
        l5 = self._first_res_block(l4,128)
        l6 = self._res_block(l5,128)
        l7 = self._first_res_block(l6,256)
        l8 = self._res_block(l7,256)
        l9 = self._first_res_block(l8,512)
        l10 = self._res_block(l9,512)
        l11 = tf.reduce_mean(l10, [1, 2])
        logits = tf.layers.dense(l11,self.class_size, kernel_initializer = tf.contrib.layers.xavier_initializer(), name = 'FC1')
        return logits


    def _init_loss(self,y,logits):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y,logits = logits,name ='loss'))
        return loss

    def _init_optimizer(self,lr,loss):
        optimizer = tf.train.AdamOptimizer(self.lr).minimize(loss)
        return optimizer


    def _first_res_block(self,x,filter_size):
        shortcut = tf.layers.conv2d(x,filter_size,[1,1],strides = 2,name = "shortcut")
        l1 = tf.layers.conv2d(x,filter_size,[3,3],strides = 2,padding = "same", activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer(),name = 'conv_1')
        l2 = tf.layers.batch_normalization(l1)
        l3 = tf.layers.conv2d(l2,filter_size,[3,3],strides = 1, padding ="same",kernel_initializer = tf.contrib.layers.xavier_initializer(),name = 'conv_2')
        l4 = tf.layers.batch_normalization(l3)
        l5 = l4 + shortcut
        return tf.nn.relu(l5)


    def _res_block(self,x,filter_size):
        shortcut = x
        l1 = tf.layers.conv2d(x,filter_size,[3,3],strides = 1, padding = "same",activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer(),name = 'conv1')
        l2 = tf.layers.batch_normalization(l1)
        l3 = tf.layers.conv2d(l1,filter_size,[3,3],strides = 1, padding = "same",activation = tf.nn.relu, kernel_initializer = tf.contrib.layers.xavier_initializer(),name = 'conv1')
        l4 = tf.layers.batch_normalization(l3)
        l5 = l4 + shortcut
        return tf.nn.relu(l5)

    def build_model(self):
        x,y = self._init_placeholders()
        logits = self._init_resnet_model(x)
        loss = self._init_loss(y,logits)
        optimizer = self._init_optimizer(lr,loss)


    def run(self,epoch):
        self.build_model()
        train_data, train_ids, test_data, test_ids = utils.read_image_names_and_assign_labels(self.class_size,20,'Images')
        perm = np.random.permutation(len(train_data))
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for i in range(epoch):
                cond = True
                while cond:
                    data,labels = utils.get_next_batch(perm,train_data,train_ids,idx,self.batch_size)
                    processed_data = prepare_img_data(data)
                    _,l = sess.run([optimizer,loss], feed_dict = {x:processed_imgs,y:train_ids[index]})
                    idx += self.batch_size
                    print("epoch: {0}, loss: {1} ".format(i,l))
                writer = tf.summary.FileWriter("output", sess.graph)
