import tensorflow as tf
from function_bank import model_fn
from function_bank import input_fn
from tensorflow.examples.tutorials.mnist import input_data
import  numpy as np
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
def main():
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(session_config=session_config)
    estimator = tf.estimator.Estimator(model_fn, 'model', config=run_config)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"images": mnist.train.images[:25600]},
        y=mnist.train.labels[:25600],
        num_epochs=1,
        batch_size=128,
        shuffle=True)
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"images": mnist.test.images[:256]},
        y=mnist.test.labels[:256],
        num_epochs=1,
        batch_size=128,
        shuffle=False)
    estimator.train(input_fn=train_input_fn, steps=30)

    #res=estimator.predict(test_input_fn)
    num = 0
    """
    for i in res:
        print(num)
        num+=1
        print(i)
    """
    return
if __name__ == '__main__':
    main()