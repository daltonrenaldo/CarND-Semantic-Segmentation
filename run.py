import tensorflow as tf
import numpy as np
from PIL import Image
import imageio
import cv2
import glob
from skvideo.io import FFmpegWriter as VideoWriter

image_shape = (160, 576)
filename = 'um_000004.png'
image_file = './data/data_road/testing/image_2/' + filename

def get_input_image(path):
    image = Image.open(path)
    print(image.size)
    yield True, np.array(image.resize((image_shape[1], image_shape[0])))

def get_video_frame(video_path):
    vidcap = cv2.VideoCapture(video_path)
    success = True

    while success:
        success, image = vidcap.read()
        if not success:
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = image[:image_shape[0]*5, 300:image_shape[1]*2 + 300]
        image = image[0:image_shape[0]*3, 0:]
        # image = cv2.resize(image, (image_shape[1], image_shape[0]))
        yield success, image

def create_mask(image):
    image = image.convert("RGBA")
    pixdata = image.load()
    width, height = image.size

    for y in range(height):
        for x in range(width):
            if pixdata[x, y] == (0, 0, 0, 255):
                pixdata[x, y] = (0, 0, 0, 0)
    return image

def get_mask_from_inference(im_softmax, image, threshold):
    segment = im_softmax[0][:, 1].reshape(image.shape[0], image.shape[1], 1)
    segment_mask = np.dot(segment > threshold, np.array([[0, 255, 0]]))
    im_mask = np.where(segment_mask, image, 0)
    im_mask = Image.fromarray(im_mask)
    im_mask = create_mask(im_mask)
    return im_mask

# video_writer = cv2.VideoWriter('segmented.avi', cv2.VideoWriter_fourcc(*'XVID'), 24, (1920, 800), True)
video_writer = VideoWriter('video_segmented.mp4')
# video_writer.open()

with tf.Session() as sess:
    # load trained model
    saver = tf.train.import_meta_graph('my_segmentation_model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    # create the graph
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name('image_input:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    logits = graph.get_tensor_by_name('fcn_logits:0')

    inputs = get_video_frame('./data/video.m4v')
    # inputs = get_input_image(image_file)
    count = 0
    for _, image in inputs:
        print(image.shape)
        # for image_file in glob.glob('./data/data_road/testing/image_2/*.png'):
        print(count)
        #     image = np.array(Image.open(image_file).resize((image_shape[1], image_shape[0])))
        feed_dict = { image_input: [image], keep_prob: 1.0 }
        #
        # # run inference
        im_softmax = sess.run([tf.nn.softmax(logits)], feed_dict)
        #
        # # extract second column (road)
        mask = get_mask_from_inference(im_softmax, image, 0.5)
        image = Image.fromarray(image)
        image.paste(mask, (0, 0), mask)
        # image = image.convert("RGB")
        image = np.array(image)
        video_writer.writeFrame(image)
        # imageio.imwrite('test' + str(count) + '.png', image)
        # image.show()
        # if (count == 5):
        #     break
        count += 1

# video_writer.stop();
# video_writer = None
