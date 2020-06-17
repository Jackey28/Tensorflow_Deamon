import tensorflow as tf
class lstmModel:
    def __init__(self,return_size):
        self.return_size = return_size
        lr = 0.001
        training_iters = 100
        self.batch_size = 128
        self.n_inputs = 28  # MNIST data input (img shape: 28*28)
        self.n_steps = 28  # time steps
        self.n_hidden_units = 128  # neurons in hidden layer
        self.n_classes = 10  # MNIST classes (0-9 digits)
    def __call__(self, inputs):
        print("in model")
 #       print(inputs['images'])
        X = tf.reshape(inputs['images'], [-1, self.n_inputs])
  #      print(X)
        X_in = tf.layers.dense(X,units=self.n_hidden_units,activation=tf.nn.relu)
        print(X_in)
        X_in = tf.reshape(X_in, [-1, self.n_steps, self.n_hidden_units])
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.n_hidden_units, forget_bias=1.0, state_is_tuple=True)
        init_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
#        outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
        outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
        print("final_state")
        print(final_state)
        print("outputs")
        print(outputs)
        print("outputs[-1]")
        print(outputs[-1])
        results = tf.layers.dense(outputs[-1],units=10,activation=tf.nn.relu)

        return results

