import tensorflow.compat.v1 as tf
import numpy as np
import Model
import LogHook
import tensorflow

tf.disable_v2_behavior()
mnist=tf.keras.datasets.mnist

def model_fn(features, labels, mode, params):
    lstmModel = Model.lstmModel(10)
    logits = lstmModel(features)
    global_step = tf.train.get_global_step()
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.reduce_max(tf.nn.softmax(logits, name="softmax_tensor"), 1)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    if mode == tf.estimator.ModeKeys.TRAIN:
        onehot_labels = tf.one_hot(labels, 10)
        loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels = onehot_labels)
        train_op = tf.train.AdamOptimizer(learning_rate=0.005)
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(onehot_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        train_hook = [LogHook.train_hooks({'loss':loss,'accuracy':accuracy, 'step': global_step})]
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          train_op=train_op.minimize(loss=loss,global_step=global_step),
                                          training_chief_hooks=train_hook)
    if mode == tf.estimator.ModeKeys.EVAL:
        onehot_labels = tf.one_hot(labels, 10)
        loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels = onehot_labels)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss)

def input_fn(total_size, isTraining):
    mnist= tensorflow.keras.datasets.mnist
    (x_train, y_train),(x_test,y_test)=mnist.load_data()
    dataset = tf.compat.v1.data.Dataset.from_tensor_slices(({'images': x_test[:total_size]}, y_test[:total_size]))
    print(x_test)
    if isTraining:
        dataset = dataset.shuffle(10 * total_size).repeat().batch(128)
    else:
        dataset = dataset.batch(128)
    print(dataset)
    return dataset.make_one_shot_iterator().get_next()


