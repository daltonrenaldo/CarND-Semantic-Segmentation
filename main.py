#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3, layer4, layer7

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    fcn8 = tf.layers.conv2d(vgg_layer7_out, filters=num_classes, kernel_size=1, padding="SAME", name='fcn8')

    fcn9 = tf.layers.conv2d_transpose(fcn8, filters=vgg_layer4_out.get_shape().as_list()[-1], kernel_size=4, strides=(2,2), padding="SAME", name='fcn9')

    fcn9_skip = tf.add(fcn9, vgg_layer4_out, name='fcn9_plus_layer4')

    fcn10 = tf.layers.conv2d_transpose(fcn9_skip, filters=vgg_layer3_out.get_shape().as_list()[-1], kernel_size=4, strides=(2,2), padding="SAME", name='fcn10')

    fcn10_skip = tf.add(fcn10, vgg_layer3_out, name='fcn10_plus_layer3')

    fcn11 = tf.layers.conv2d_transpose(fcn10_skip, filters=num_classes, kernel_size=16, strides=(8,8), padding="SAME", name='fcn11')

    return fcn11

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # Reshape 4D tensors to 2D, each row represents a pixel, each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name="fcn_logits")
    correct_label_reshaped = tf.reshape(correct_label, (-1, num_classes))

    # Calculate distance from actual labels using cross entropy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label_reshaped[:])
    # Take mean for total loss
    loss_op = tf.reduce_mean(cross_entropy, name="fcn_loss")

    # The model implements this operation to find the weights/parameters that would yield correct pixel labels
    train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name="fcn_train_op")

    return logits, train_op, loss_op

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    keep_prob_value = 0.5
    learning_rate_value = 0.001
    for epoch in range(epochs):
        # Create function to get batches
        total_loss = 0
        for X_batch, gt_batch in get_batches_fn(batch_size):

            loss, _ = sess.run([cross_entropy_loss, train_op],
            feed_dict={input_image: X_batch, correct_label: gt_batch,
            keep_prob: keep_prob_value, learning_rate:learning_rate_value})

            total_loss += loss;

        print("EPOCH {} ...".format(epoch + 1))
        print("Loss = {:.3f}".format(total_loss))
        print()
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    epochs = 40
    batch_size=15
    dropout = 0.75
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
    learning_rate = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)

        print("loadded vgg")

        model_output = layers(layer3, layer4, layer7, num_classes)

        print("got output")

        logits, train_op, cross_entropy_loss = optimize(model_output, correct_label, learning_rate, num_classes)

        print("optimizer configured")
        saver = tf.train.Saver()

        sess.run(tf.compat.v1.global_variables_initializer())
        print("global variables loaded")
        sess.run(tf.local_variables_initializer())

        print("Model build successful, starting training")

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input, correct_label, keep_prob, learning_rate)

        saver.save(sess, 'my_segmentation_model')
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        print('Done')

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
