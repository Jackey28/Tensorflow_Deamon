import tensorflow as tf
import numpy as np
import Model
import LogHook
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
def input_fn(isTraining,batch_size):
 #   xs, ys = mnist.train.next_batch(batch_size=128)
    train_xs = mnist.train.images[:3000]
    #train_xs = np.reshape(train_xs,[-1,28,28])
    train_ys = mnist.train.labels[:3000]
    test_xs = mnist.test.images[:300]
    #test_xs = np.reshape(test_xs,[-1,28,28])
    test_ys = mnist.test.labels[:300]
    train_ys = np.array(train_ys)
    train_xs = np.array(train_xs)

    dataset = tf.data.Dataset.from_tensor_slices(({
        'inputs': train_xs, "labels":train_ys}))
    #dataset = tf.data.Dataset.from_tensor_slices((train_xs, train_ys))
    #dataset = tf.data.Dataset.from_tensor_slices(tf.random_uniform([100,2]))
    if isTraining:
        train_xs = tf.convert_to_tensor(train_xs)
        train_ys = tf.convert_to_tensor(train_ys)
        #return train_xs,train_ys
        dataset = dataset.shuffle(10 * 10000).repeat().batch(batch_size=batch_size)
        #dataset = dataset.shuffle(10 * total_size).batch(batch_size)
    else:
        dataset = tf.data.Dataset.from_tensor_slices(({'inputs': test_xs}, test_ys))
      #  return test_xs,test_ys
 #       dataset = dataset.batch(batch_size=batch_size)
    return dataset.make_one_shot_iterator().get_next()

def model_fn(features, labels, mode, params):
    print("in model  fn ")
    lstmModel = Model.lstmModel(10)
    logits = lstmModel(features)
    print("logits==")
    print(logits)
    global_step = tf.train.get_global_step()
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),  # 选择logits中最大的
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    if mode == tf.estimator.ModeKeys.TRAIN:
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        #a = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        train_op = tf.train.AdamOptimizer(learning_rate=0.01)
  #      train_op = tf.train.MomentumOptimizer(learning_rate=0.1,momentum=0.9)#LJ
  #      train_op = tf.train.AdagradOptimizer(learning_rate=0.1)#LJ
        #train_op = tf.train.RMSPropOptimizer(0.01)#LJ
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        train_hook = [LogHook.train_hooks({'cost':cost,'accuracy':accuracy})]

        return tf.estimator.EstimatorSpec(mode=mode, loss=cost,
                                          train_op=train_op.minimize(loss=cost,global_step=global_step),
                                          training_chief_hooks=train_hook)
                                        #  training_chief_hooks=train_hooks)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=cost,)
#input_fn(isTraining=True,batch_size=128)
if __name__ == '__main__':
    input_fn()