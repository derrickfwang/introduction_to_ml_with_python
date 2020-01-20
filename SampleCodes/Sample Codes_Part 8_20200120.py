#########################################################################################
#############                         Other Sample Codes                    #############
#########################################################################################

# Part 8: Deep learning, time series

# 1. LSTM with keras
df1=df1.set_index(pd.DatetimeIndex(df1['Date'])) 
sns.lineplot(data = df, x = "X", y = "y")
from sklearn.metrics import mean_squared_error, mean_absolute_error

from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from keras.optimizers import RMSprop
model = Sequential()
'''The input to an LSTM layer has to be 3D tensor. The default LSTM layer output is 2D '''
model.add(LSTM(32, input_shape=(NUM_TIMESTEPS,features )))## The shape of each input sample is defined in 1st layer only
model.add(Dense(output))
model.compile(optimizer=RMSprop(), loss='mae')
model.summary()

history=model.fit(train_X_seq,train_y_seq,epochs=500,shuffle=False,
					batch_size=BATCH_SIZE,
					validation_data=(val_X_seq,val_y_seq))
model.predict(test_X_seq)

# 2. Tensorflow
# https://adventuresinmachinelearning.com/python-tensorflow-tutorial/
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Python optimisation variables
learning_rate = 0.5
epochs = 10
batch_size = 100

# declare the training data placeholders
# input x - for 28 x 28 pixels = 784
x = tf.placeholder(tf.float32, [None, 784])
# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, 10])

# now declare the weights connecting the input to the hidden layer
W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([300]), name='b1')
# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([10]), name='b2')

# calculate the output of the hidden layer
hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.relu(hidden_out)
# now calculate the hidden layer output - in this case, let's use a softmax activated
# output layer
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)   # loss function
                         + (1 - y) * tf.log(1 - y_clipped), axis=1))

# add an optimiser
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# start the session
with tf.Session() as sess:
   # initialise the variables
   sess.run(init_op)
   total_batch = int(len(mnist.train.labels) / batch_size)
   for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
             _, c = sess.run([optimiser, cross_entropy], 
                         feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
   print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
   
   
# 3. Time-series
from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams
rcParams['figure.figsize'] = 12, 8
series = df2.Amt
result = seasonal_decompose(series, model='additive'
                            #, freq= 8, extrapolate_trend = 30
                           )
fig = result.plot()
plt.show()


# 4. statsmodels 
from statsmodels.tsa.seasonal import seasonal_decompose  # time series data

# lr model from statsmodels
import statsmodels.api as sm
x = np.linspace(0, 10, nsample)
# 使用 sm.add_constant() 在 array 上加入一列常项1
X = sm.add_constant(x)

beta = np.array([1, 10])
e = np.random.normal(size=nsample)
y = np.dot(X, beta) + e

model = sm.OLS(y,X)
results = model.fit()
print(results.summary())

results.predict(X_test)





