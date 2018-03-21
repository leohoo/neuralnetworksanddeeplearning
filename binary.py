import tensorflow as tf
import numpy as np

def hotbit(i):
  return np.identity(10)[i,:]

def binary(n):
    b = np.zeros(np.array(n).shape + (4,))
    for i in range(4):
        b[:, 3-i] = (np.array(n)>>i) & 1

    return b

x = tf.placeholder(tf.float32, [None, 10])
y = tf.placeholder(tf.float32, [None, 4])

N = 20

W = tf.Variable(tf.zeros([10, 4]))
# b = tf.Variable(tf.zeros([4]))

prediction = tf.matmul(x, W)
loss = tf.reduce_mean(tf.square(y-prediction))
train_step = tf.train.GradientDescentOptimizer(0.3).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(2000):
        n = np.random.randint(0,10, N)
        noise = (np.ones([10,10]) - np.identity(10))[:,n] * np.random.rand(10,N)*0.01
        x_data = hotbit(n) # * (np.random.rand(N)*0.01 + 0.99) + noise
        y_data = binary(n)
        sess.run(train_step, feed_dict={x:x_data, y:y_data})

    print(sess.run(W))
    w = sess.run(W)
    for i in range(10):
        print("loss(%d) =" % i, sess.run(loss, feed_dict={x:hotbit([i]), y:binary([i])}))
        print(i, sess.run(prediction, feed_dict={x:hotbit([i])}))
        print("%d * W" % i, np.matmul(hotbit([i]), w))
