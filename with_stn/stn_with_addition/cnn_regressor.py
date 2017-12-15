#Tajwar Abrar Aleef
#CNN-simple regression problem

import tensorflow as tf
import tensorlayer as tl
from stn_transform import stn_affine
from tensorlayer.layers import *
from scipy import misc


LOGDIR = 'D:\LEIDEN\REPORT\stn_included/Noisy_image\LOG'
imagespath = 'D:\LEIDEN\Applicator\Proper_Noise_negtrans_3/'
sess = tf.InteractiveSession()

#Toggle train on and off
train=1;
#Hyper Parameters
batch_size = 200
dim1 = 64
dim2 = 64
n_epoch = 100
learning_rate = 0.0001
print_freq = 10

#loading ground truth 
csv = np.loadtxt(open("D:\LEIDEN\Applicator\Proper_Noise_negtrans_3\Proper_Noise_negtrans_3.csv", "rb"), delimiter=",")
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
        ln = Conv2d(nin, 16, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='tc01')
        ln = Conv2d(nin, 64, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc02')
        ln = Conv2d(ln, 128, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc03')
        ln = FlattenLayer(ln, name='f01')
        ln = DenseLayer(ln, n_units=64, act=tf.identity, name='d01')
        ln = DenseLayer(ln, n_units=3, act=tf.identity, name='d02')

        #STN
        stn = stn_affine(nin, ln, out_size=[dim1, dim2], name='ST')


        nt = Conv2d(stn, 16, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='tc1')
        nt = Conv2d(nt, 32, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc2')
        # 32x32x16

        nt = Conv2d(nt, 64, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc3')
        nt = Conv2d(nt, 128, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc4')

        nt = FlattenLayer(nt, name='f')
        nt = DenseLayer(nt, n_units=64, act=tf.identity, name='d1')
        nt = DenseLayer(nt, n_units=3, act=tf.identity, name='do')

        y = nt.outputs
        localization_out= ln.outputs
        final_out = tf.add(localization_out, y)

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

    return nt, cost,  final_out, loss1, loss2 , loss3, localization_out

net_train,  cost, y_temp, ls1, ls2, ls3, pr1 = model(x, is_train=True, reuse=False)
net_test,  cost_test, y_temp_test, lsz1, lsz2, lsz3, pr2 = model(x, is_train=False, reuse=True)

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
net_train.print_params()
net_train.print_layers()

if(train ==1):
    for epoch in range(n_epoch):
        start_time = time.time()

        for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
            sess.run(train_op, feed_dict={x: X_train_a, y_: y_train_a})

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
            train_loss, train_loss1, train_loss2, train_loss3, n_batch = 0, 0, 0, 0 , 0

            for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=False):
               sum, err, lss1,lss2,lss3= sess.run([merged, cost_test, lsz1, lsz2, lsz3], feed_dict={x: X_train_a, y_: y_train_a})
               train_writer.add_summary(sum, epoch)
               train_loss = train_loss+ err; n_batch = n_batch + 1; train_loss1 += lss1 ;train_loss2 += lss2 ;train_loss3 += lss3;

            print("   train loss: %f" % (train_loss/ n_batch))
            print("   rot loss: %f" % (train_loss1 / n_batch))
            print("   trans1 loss: %f" % (train_loss2 / n_batch))
            print("   trans2 loss: %f" % (train_loss3 / n_batch))

            val_loss, val_loss1, val_loss2, val_loss3, n_batch = 0, 0, 0, 0, 0
            for X_val_a, y_val_a in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=False):
                 summary, err, lss1,lss2,lss3, parameter, yyy = sess.run([merged, cost_test, lsz1, lsz2, lsz3, pr2, y_temp_test], feed_dict={x: X_val_a, y_: y_val_a})
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

