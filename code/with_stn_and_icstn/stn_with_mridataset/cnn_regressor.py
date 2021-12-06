# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 16:22:29 2017

@author: Tajwar Abrar Aleef
"""

#Tajwar Abrar Aleef
#CNN regression with sample as MRI background

import tensorflow as tf
import tensorlayer as tl
from stn_transform import stn_affine
from tensorlayer.layers import *
from scipy import misc
import numpy as np
from sklearn.cross_validation import train_test_split


LOGDIR = '/home/tempuser/MRI_FINAL\LOG/'
imagespath = '/home/tempuser/Proper_MRI_Huge/'
sess = tf.InteractiveSession()

#Toggle train on and off
train=0;
batch_size =200
dim1 = 90
dim2 = 160
n_epoch = 150
learning_rate = 0.0001
print_freq = 1
save_freq = 20
print("start")
#loading ground truth 
csv = np.loadtxt(open("/home/tempuser/Proper_MRI_Huge/Proper_MRI_Huge.csv", "rb"), delimiter=",")
y=csv[0:150000]

#loading training, validation and testing set
X = np.empty([150000,dim1,dim2,1], dtype= np.float)
for num in range(0, 150000):
    image= misc.imread(imagespath + str(num+1) + '.png')
    image = misc.imresize(image, (dim1, dim2))
    X[num,:,:,0]=image / 255
print("loaded")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

#Model definition

x = tf.placeholder(tf.float32, shape=[batch_size, dim1, dim2, 1], name='x')
y_ = tf.placeholder(tf.float32, shape=[batch_size, 3], name='y_')
gamma_init = tf.random_normal_initializer(1., 0.02)

def model(x, is_train, reuse):
    with tf.variable_scope("STN", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        nin = InputLayer(x, name='in')

        #Localization Network
        ln1 = Conv2d(nin, 16, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='tc01')
        
        ln1 = Conv2d(ln1, 32, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc02')
             
            
        ln1 = Conv2d(ln1, 64, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc03')
        ln1 = BatchNormLayer(ln1, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='batch_norm1')
        ln1 = Conv2d(ln1, 128, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc04')
        ln1 = BatchNormLayer(ln1, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='batch_norm2')
        ln1 = FlattenLayer(ln1, name='f01')
        ln1 = DenseLayer(ln1, n_units=64, act=tf.identity, name='d01')
        ln1 = DenseLayer(ln1, n_units=3, act=tf.identity, name='d02')

        stn = stn_affine(nin, ln1, out_size=[dim1, dim2], name='ST1')

        ln2 = Conv2d(stn, 16, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='tc05')
        ln2 = Conv2d(ln2, 32, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc06')
        ln2 = BatchNormLayer(ln2, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='batch_norm3')
        ln2 = Conv2d(ln2, 64, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc07')
        ln2 = BatchNormLayer(ln2, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='batch_norm4')
        ln2 = Conv2d(ln2, 128, (2, 2), (2, 2), act=tf.nn.relu, padding='SAME', name='tc08')
        ln2 = FlattenLayer(ln2, name='f02')
        ln2 = DenseLayer(ln2, n_units=64, act=tf.identity, name='d03')
        ln2 = DenseLayer(ln2, n_units=3, act=tf.identity, name='d04')

        final_out = tf.add(ln1.outputs,ln2.outputs )

		#overall mean error of all the three parameters
        ce = tl.cost.mean_squared_error(final_out, y_, 'cost')
        L2 = 0
        for p in tl.layers.get_variables_with_name('relu/W', True, True):
            L2 += tf.contrib.layers.l2_regularizer(0.004)(p)
        cost = ce + L2
        tf.summary.scalar('cost',cost)

        #Individual errors for rotation, x translation and y translation
        loss1 = tf.losses.mean_squared_error(final_out[:,0], y_[:,0])
        tf.summary.scalar('loss1', loss1)

        loss2 = tf.losses.mean_squared_error(final_out[:,1], y_[:,1])
        tf.summary.scalar('loss2', loss2)

        loss3 = tf.losses.mean_squared_error(final_out[:,2], y_[:,2])
        tf.summary.scalar('loss3', loss3)

    return ln1,ln2, cost,  final_out, loss1, loss2 , loss3, ln1.outputs

net_train1,net_train2,  cost, y_temp, ls1, ls2, ls3, pr1 = model(x, is_train=True, reuse=False)
net_test1,net_test2,  cost_test, y_temp_test, lsz1, lsz2, lsz3, pr2 = model(x, is_train=False, reuse=True)

train_params = tl.layers.get_variables_with_name('STN', train_only=True, printable=True)
train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
    epsilon=1e-05, use_locking=False).minimize(cost, var_list=train_params)

#for tensorboard visualization
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(LOGDIR + '/train',
                                      sess.graph)
test_writer = tf.summary.FileWriter(LOGDIR + '/test')
tf.global_variables_initializer().run()

#Training and printing losses at every 10 epochs
tl.layers.initialize_global_variables(sess)
# net_train.print_params()
# net_train.print_layers()

if(train ==1):
    for epoch in range(n_epoch):
        start_time = time.time()

        for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
            feedict = {x: X_train_a, y_: y_train_a}
         #   feedict.update(net_train1.all_drop)
          #  feedict.update(net_train2.all_drop)
           # feedict.update(net_train3.all_drop)
            #feedict.update(net_train4.all_drop)

            sess.run(train_op, feed_dict=feedict)

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
                train_loss, train_loss1, train_loss2, train_loss3, n_batch = 0, 0, 0, 0 , 0

                for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=False):
             #       dp_dict1 = tl.utils.dict_to_one(net_test1.all_drop)
              #      dp_dict2 = tl.utils.dict_to_one(net_test2.all_drop)
             #       dp_dict3 = tl.utils.dict_to_one(net_test3.all_drop)
                    #dp_dict4 = tl.utils.dict_to_one(net_test4.all_drop)
                    feedict = {x: X_train_a, y_: y_train_a}
           #         feedict.update(dp_dict1)
            #        feedict.update(dp_dict2)
            #        feedict.update(dp_dict3)
                    #feedict.update(dp_dict4)

                    sum, err, lss1,lss2,lss3= sess.run([merged, cost_test, lsz1, lsz2, lsz3], feed_dict=feedict)
                    train_writer.add_summary(sum, epoch)
                    train_loss = train_loss+ err; n_batch = n_batch + 1; train_loss1 += lss1 ;train_loss2 += lss2 ;train_loss3 += lss3;

                print("   train loss: %f" % (train_loss/ n_batch))
                print("   rot loss: %f" % (train_loss1 / n_batch))
                print("   trans1 loss: %f" % (train_loss2 / n_batch))
                print("   trans2 loss: %f" % (train_loss3 / n_batch))

                val_loss, val_loss1, val_loss2, val_loss3, n_batch = 0, 0, 0, 0, 0
                for X_val_a, y_val_a in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=False):
                  #   dp_dict1 = tl.utils.dict_to_one(net_test1.all_drop)
                 #    dp_dict2 = tl.utils.dict_to_one(net_test2.all_drop)
              #       dp_dict3 = tl.utils.dict_to_one(net_test3.all_drop)
                     #dp_dict4 = tl.utils.dict_to_one(net_test4.all_drop)
                     feedict = {x: X_val_a, y_: y_val_a}
                #     feedict.update(dp_dict1)
               #      feedict.update(dp_dict2)
               #      feedict.update(dp_dict3)
                     #feedict.update(dp_dict4)
                     summary, err, lss1,lss2,lss3, parameter, yyy = sess.run([merged, cost_test, lsz1, lsz2, lsz3, pr2, y_temp_test], feed_dict=feedict)
                     test_writer.add_summary(summary, epoch)
                     val_loss += err; val_loss1 += lss1; val_loss2 += lss2; val_loss3 += lss3; n_batch += 1
                print("   val loss: %f" % (val_loss/ n_batch))
                print("   rot loss: %f" % (val_loss1 / n_batch))
                print("   trans1 loss: %f" % (val_loss2 / n_batch))
                print("   trans2 loss: %f" % (val_loss3 / n_batch))

                if epoch + 1 == 1 or (epoch + 1) % save_freq == 0:
                    tl.files.save_npz(net_train2.all_params, name='model_noise_stn.npz')

if(train==1):
    tl.files.save_npz(net_train2.all_params, name='model_noise_stn.npz')
if(train==0):
    load_params = tl.files.load_npz(path='', name='model_noise_stn.npz')
    tl.files.assign_params(sess, load_params, net_test2)

print('Evaluation')

# y_out1 = np.zeros([y_test.shape[0],3], dtype= np.float)
# err1 = np.zeros([y_test.shape[0],4], dtype= np.float)
# errt=[]
# testls1t=[]
# testls2t=[]
# testls3t=[]
n=0
y_outt=np.zeros([batch_size,3], dtype= np.float)
for X_test_a, y_test_a in tl.iterate.minibatches(
                             X_test, y_test, batch_size, shuffle=False):
    err,  y_out, testls1, testls2, testls3 = sess.run([ cost_test,  y_temp_test, lsz1, lsz2, lsz3], feed_dict={x: X_test_a, y_: y_test_a})
    # errt= np.append(errt,err)
    # testls1t = np.append(testls1t, testls1)
    # testls2t = np.append(testls2t, testls2)
    # testls3t = np.append(testls3t, testls3)
    if(n==0):
        y_outt = y_out

    if(n==1):
        y_outt = np.append(y_outt,y_out, axis=0)
    n = 1
# err1[:,0] = errt
# err1[:, 1] = testls1t
# err1[:, 2] = testls2t
# err1[:, 3] = testls3t




# #Saving Evaluation Data

np.save("X_test.npy",X_test);
np.savetxt("Y_MRI_in.csv", y_test, delimiter=",")
np.savetxt("Y_MRI_out.csv", y_outt, delimiter=",")
# np.savetxt("binary_error_stn.csv", err1, delimiter=",")

