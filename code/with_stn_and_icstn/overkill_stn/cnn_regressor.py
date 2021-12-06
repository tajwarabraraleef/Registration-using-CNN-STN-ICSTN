#Tajwar Abrar Aleef
#With Overkill STN set up regression problem

import tensorflow as tf
import tensorlayer as tl
from stn_transform import stn_affine
from tensorlayer.layers import *
from scipy import misc


LOGDIR = 'D:\LEIDEN\REPORT\comparison_250epoch\Overkill\LOG'
imagespath = 'D:\LEIDEN\Applicator\Proper_Noise_negtrans/'
sess = tf.InteractiveSession()

#Toggle train on and off
train=1;
batch_size = 200
dim1 = 64
dim2 = 64
n_epoch = 150
learning_rate = 0.0001
print_freq = 10

#loading ground truth 
csv = np.loadtxt(open("D:\LEIDEN\Applicator\Proper_Noise_negtrans\Proper_Noise_negtrans.csv", "rb"), delimiter=",")
y_train = csv[0:8000]
y_val = csv[8000:9000]
y_test = csv[9000:10000]

#loading training, validation and testing set
X_train = np.empty([8000,dim1,dim2,1], dtype= np.float)
for num in range(0, 8000):
    image= misc.imread(imagespath + str(num+1) + '.png')
    X_train[num,:,:,0]=image / 255

X_val = np.empty([1000,dim1,dim2,1], dtype= np.float)
for num in range(0, 1000):
    image= misc.imread(imagespath + str(num+8001) + '.png')
    X_val[num,:,:,0]=image/255

X_test = np.empty([1000,dim1,dim2,1], dtype= np.float)
for num in range(0, 1000):
    image= misc.imread(imagespath + str(num+9001) + '.png')
    X_test[num,:,:,0]=image/255


#Model definition 

x = tf.placeholder(tf.float32, shape=[batch_size, dim1, dim2, 1], name='x')
y_ = tf.placeholder(tf.float32, shape=[batch_size, 3], name='y_')


def model(x, is_train, reuse):
    with tf.variable_scope("STN", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        nin = InputLayer(x, name='in')

        #Localization Network


        ln1 = Conv2d(nin, 16, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='tc01')
        ln1 = Conv2d(ln1, 64, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc02')
        ln1 = Conv2d(ln1, 128, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc03')
        ln1 = FlattenLayer(ln1, name='f01')
        ln1 = DropoutLayer(ln1, 0.5, True, is_train, name='drop1')
        ln1 = DenseLayer(ln1, n_units=64, act=tf.identity, name='d01')
        ln1 = DropoutLayer(ln1, 0.5, True, is_train, name='drop2')
        ln1 = DenseLayer(ln1, n_units=3, act=tf.identity, name='d02')

        stn = stn_affine(nin, ln1, out_size=[dim1, dim2], name='ST1')

        ln2 = Conv2d(stn, 16, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='tc04')
        ln2 = Conv2d(ln2, 64, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc05')
        ln2 = Conv2d(ln2, 128, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc06')
        ln2 = FlattenLayer(ln2, name='f02')
        ln2 = DropoutLayer(ln2, 0.5, True, is_train, name='drop3')
        ln2 = DenseLayer(ln2, n_units=64, act=tf.identity, name='d03')
        ln2 = DropoutLayer(ln2, 0.5, True, is_train, name='drop4')
        ln2 = DenseLayer(ln2, n_units=3, act=tf.identity, name='d04')

        stn2 = stn_affine(stn, ln2, out_size=[dim1, dim2], name='ST2')

        ln3 = Conv2d(stn2, 16, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='tc07')
        ln3 = Conv2d(ln3, 64, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc08')
        ln3 = Conv2d(ln3, 128, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc09')
        ln3 = FlattenLayer(ln3, name='f03')
        ln3 = DropoutLayer(ln3, 0.5, True, is_train, name='drop5')
        ln3 = DenseLayer(ln3, n_units=64, act=tf.identity, name='d05')
        ln3 = DropoutLayer(ln3, 0.5, True, is_train, name='drop6')
        ln3 = DenseLayer(ln3, n_units=3, act=tf.identity, name='d06')

        stn3 = stn_affine(stn2, ln3, out_size=[dim1, dim2], name='ST3')

        ln4 = Conv2d(stn3, 16, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='tc10')
        ln4 = Conv2d(ln4, 64, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc11')
        ln4 = Conv2d(ln4, 128, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc12')
        ln4 = FlattenLayer(ln4, name='f04')
        ln4 = DropoutLayer(ln4, 0.5, True, is_train, name='drop7')
        ln4 = DenseLayer(ln4, n_units=64, act=tf.identity, name='d07')
        ln4 = DropoutLayer(ln4, 0.5, True, is_train, name='drop8')
        ln4 = DenseLayer(ln4, n_units=3, act=tf.identity, name='d08')

        final_out = tf.add(ln1.outputs,ln2.outputs )
        final_out = tf.add(final_out,ln3.outputs)
        final_out = tf.add(final_out,ln4.outputs)

		#overall mean error of all the three parameters
        cost = tl.cost.mean_squared_error(final_out, y_, 'cost')
        tf.summary.scalar('cost',cost)

        #Individual errors for rotation, x translation and y translation
        loss1 = tf.losses.mean_squared_error(final_out[:,0], y_[:,0])
        tf.summary.scalar('loss1', loss1)

        loss2 = tf.losses.mean_squared_error(final_out[:,1], y_[:,1])
        tf.summary.scalar('loss2', loss2)

        loss3 = tf.losses.mean_squared_error(final_out[:,2], y_[:,2])
        tf.summary.scalar('loss3', loss3)

    return ln1,ln2,ln3,ln4, cost,  final_out, loss1, loss2 , loss3, ln1.outputs

net_train1,net_train2,net_train3,net_train4,  cost, y_temp, ls1, ls2, ls3, pr1 = model(x, is_train=True, reuse=False)
net_test1,net_test2,net_test3,net_test4,  cost_test, y_temp_test, lsz1, lsz2, lsz3, pr2 = model(x, is_train=False, reuse=True)

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
            feedict.update(net_train1.all_drop)
            feedict.update(net_train2.all_drop)
            feedict.update(net_train3.all_drop)
            feedict.update(net_train4.all_drop)

            sess.run(train_op, feed_dict=feedict)

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
                train_loss, train_loss1, train_loss2, train_loss3, n_batch = 0, 0, 0, 0 , 0

                for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=False):
                    dp_dict1 = tl.utils.dict_to_one(net_test1.all_drop)
                    dp_dict2 = tl.utils.dict_to_one(net_test2.all_drop)
                    dp_dict3 = tl.utils.dict_to_one(net_test3.all_drop)
                    dp_dict4 = tl.utils.dict_to_one(net_test4.all_drop)
                    feedict = {x: X_train_a, y_: y_train_a}
                    feedict.update(dp_dict1)
                    feedict.update(dp_dict2)
                    feedict.update(dp_dict3)
                    feedict.update(dp_dict4)

                    sum, err, lss1,lss2,lss3= sess.run([merged, cost_test, lsz1, lsz2, lsz3], feed_dict=feedict)
                    train_writer.add_summary(sum, epoch)
                    train_loss = train_loss+ err; n_batch = n_batch + 1; train_loss1 += lss1 ;train_loss2 += lss2 ;train_loss3 += lss3;

                print("   train loss: %f" % (train_loss/ n_batch))
                print("   rot loss: %f" % (train_loss1 / n_batch))
                print("   trans1 loss: %f" % (train_loss2 / n_batch))
                print("   trans2 loss: %f" % (train_loss3 / n_batch))

                val_loss, val_loss1, val_loss2, val_loss3, n_batch = 0, 0, 0, 0, 0
                for X_val_a, y_val_a in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=False):
                     dp_dict1 = tl.utils.dict_to_one(net_test1.all_drop)
                     dp_dict2 = tl.utils.dict_to_one(net_test2.all_drop)
                     dp_dict3 = tl.utils.dict_to_one(net_test3.all_drop)
                     dp_dict4 = tl.utils.dict_to_one(net_test4.all_drop)
                     feedict = {x: X_val_a, y_: y_val_a}
                     feedict.update(dp_dict1)
                     feedict.update(dp_dict2)
                     feedict.update(dp_dict3)
                     feedict.update(dp_dict4)
                     summary, err, lss1,lss2,lss3, parameter, yyy = sess.run([merged, cost_test, lsz1, lsz2, lsz3, pr2, y_temp_test], feed_dict=feedict)
                     test_writer.add_summary(summary, epoch)
                     val_loss += err; val_loss1 += lss1; val_loss2 += lss2; val_loss3 += lss3; n_batch += 1
                print("   val loss: %f" % (val_loss/ n_batch))
                print("   rot loss: %f" % (val_loss1 / n_batch))
                print("   trans1 loss: %f" % (val_loss2 / n_batch))
                print("   trans2 loss: %f" % (val_loss3 / n_batch))
#evaluation

if(train==1):
    tl.files.save_npz(net_train.all_params, name='model_noise_stn.npz')
if(train==0):
    load_params = tl.files.load_npz(path='', name='model_noise_stn.npz')
    tl.files.assign_params(sess, load_params, net_test)

print('Evaluation')


n=0
y_outt=np.zeros([batch_size,3], dtype= np.float)
for X_test_a, y_test_a in tl.iterate.minibatches(
                             X_test, y_test, batch_size, shuffle=False):
    err,  y_out, testls1, testls2, testls3 = sess.run([ cost_test,  y_temp_test, lsz1, lsz2, lsz3], feed_dict={x: X_test_a, y_: y_test_a})
    if(n==0):
        y_outt = y_out

    if(n==1):
        y_outt = np.append(y_outt,y_out, axis=0)
    n = 1



# #Saving Evaluation Data
np.savetxt("noise_y_stn_out.csv", y_outt, delimiter=",")
# np.savetxt("binary_error_stn.csv", err1, delimiter=",")

