#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from function_bank import model_fn
from function_bank import input_fn
import tensorflow
import  numpy as np

mnist= tensorflow.keras.datasets.mnist
(x_train, y_train),(x_test,y_test)=mnist.load_data()
training_data_size = 128*128*2
testing_data_size = 128*10
def main():
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(session_config=session_config)
    estimator = tf.estimator.Estimator(model_fn, 'model', config=run_config)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"images": x_train[:training_data_size]},
        y=y_train[:training_data_size],
        num_epochs=1,
        batch_size=128,
        shuffle=True)

    test_input_fn = lambda: input_fn(testing_data_size, False)

    estimator.train(input_fn=train_input_fn, max_steps=128*10)
    res = estimator.predict(test_input_fn)

    for i in res:
        print("value: ", i['classes'], ", with probabilities: ", i['probabilities'])
if __name__ == '__main__':
    main()
