import tensorflow as tf
import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1' # 默认，显示所有信息
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # 只显示 warning 和 Error
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3' # 只显示 Error



##################测试用例（官网给出的）####################################
# mnist = tf.keras.datasets.mnist
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
#
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10, activation='softmax')
# ])
#
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=5)
#
# model.evaluate(x_test,  y_test, verbose=2)


########实现一个加法运算##########################


a = tf.constant(5.0)

b = tf.constant(6.0)


# print(a,b)

# print(tf.add(a,b))

########################求解 1 + 1/2 + 1/2^2 + 1/2^3 + ... + 1/2^50$$$$$$$$$$$$$$$$$
print(tf.__version__)

x = tf.constant(0.)
y = tf.constant(1.)

for iteration in range(50):
	x = x + y
	y = y / 2

print(x.numpy())
