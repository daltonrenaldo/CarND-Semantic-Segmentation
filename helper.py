'''
You should not edit helper.py as part of your submission.

This file is used primarily to download vgg if it has not yet been,
give you the progress of the download, get batches for your training,
as well as around generating and saving the image outputs.
'''

import re
import random
import numpy as np
import os.path
from PIL import Image
import imageio
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm


class DLProgress(tqdm):
	"""
	Report download progress to the terminal.
	:param tqdm: Information fed to the tqdm library to estimate progress.
	"""
	last_block = 0

	def hook(self, block_num=1, block_size=1, total_size=None):
		"""
		Store necessary information for tracking progress.
		:param block_num: current block of the download
		:param block_size: size of current block
		:param total_size: total download size, if known
		"""
		self.total = total_size
		self.update((block_num - self.last_block) * block_size)  # Updates progress
		self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
	"""
	Download and extract pretrained vgg model if it doesn't exist
	:param data_dir: Directory to download the model to
	"""
	vgg_filename = 'vgg.zip'
	vgg_path = os.path.join(data_dir, 'vgg')
	vgg_files = [
		os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
		os.path.join(vgg_path, 'variables/variables.index'),
		os.path.join(vgg_path, 'saved_model.pb')]

	missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
	if missing_vgg_files:
		# Clean vgg dir
		if os.path.exists(vgg_path):
			shutil.rmtree(vgg_path)
		os.makedirs(vgg_path)

		# Download vgg
		print('Downloading pre-trained vgg model...')
		with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
			urlretrieve(
				'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
				os.path.join(vgg_path, vgg_filename),
				pbar.hook)

		# Extract vgg
		print('Extracting model...')
		zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
		zip_ref.extractall(data_dir)
		zip_ref.close()

		# Remove zip file to save space
		os.remove(os.path.join(vgg_path, vgg_filename))


def gen_batch_function(data_folder, image_shape):
	"""
	Generate function to create batches of training data
	:param data_folder: Path to folder that contains all the datasets
	:param image_shape: Tuple - Shape of image
	:return:
	"""
	def get_batches_fn(batch_size):
		"""
		Create batches of training data
		:param batch_size: Batch Size
		:return: Batches of training data
		"""
		# Grab image and label paths
		image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
		label_paths = {
			re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
			for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
		background_color = np.array([255, 0, 0])

		# Shuffle training data
		random.shuffle(image_paths)
		# Loop through batches and grab images, yielding each batch
		for batch_i in range(0, len(image_paths), batch_size):
			images = []
			gt_images = []
			for image_file in image_paths[batch_i:batch_i+batch_size]:
				gt_image_file = label_paths[os.path.basename(image_file)]
				# Re-size to image_shape
				image = np.array(Image.open(image_file).resize((image_shape[1], image_shape[0])))
				gt_image = np.array(Image.open(gt_image_file).resize((image_shape[1], image_shape[0])))

				# Create "one-hot-like" labels by class
				gt_bg = np.all(gt_image == background_color, axis=2)
				gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
				gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

				images.append(image)
				gt_images.append(gt_image)

			yield np.array(images), np.array(gt_images)
	return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
	"""
	Generate test output using the test images
	:param sess: TF session
	:param logits: TF Tensor for the logits
	:param keep_prob: TF Placeholder for the dropout keep probability
	:param image_pl: TF Placeholder for the image placeholder
	:param data_folder: Path to the folder that contains the datasets
	:param image_shape: Tuple - Shape of image
	:return: Output for for each test image
	"""
	for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
		image = np.array(Image.open(image_file).resize((image_shape[1], image_shape[0])))

		# Run inference
		im_softmax = sess.run(
			[tf.nn.softmax(logits)],
			{keep_prob: 1.0, image_pl: [image]})
		# print('=========softmax===============')
		# print(im_softmax)
		# print('========SEGMENTATION===========')
		# Splice out second column (road), reshape output back to image_shape
		im_softmax1 = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
		im_softmax2 = im_softmax[0][:, 0].reshape(image_shape[0], image_shape[1])
		# If road softmax > 0.5, prediction is road
		segmentation_road = (im_softmax1 > 0.5).reshape(image_shape[0], image_shape[1], 1)
		segmentation_bg   = (im_softmax2 > 0.5).reshape(image_shape[0], image_shape[1], 1)
		# print(segmentation)
		# Create mask based on segmentation to apply to original image
		mask = np.dot(segmentation_bg, np.array([[0, 255, 0, 127]]))
		street_im = np.where(segmentation_road, image, mask)
		# mask = mask.reshape(image_shape[0], image_shape[1])
		# mask = Image.fromarray(mask, mode="RGBA")
		# street_im = Image.fromarray(image)
		# street_im.paste(mask, mask=mask)
		# street_im = Image.fromarray(street_im)

		yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
	"""
	Save test images with semantic masks of lane predictions to runs_dir.
	:param runs_dir: Directory to save output images
	:param data_dir: Path to the directory that contains the datasets
	:param sess: TF session
	:param image_shape: Tuple - Shape of image
	:param logits: TF Tensor for the logits
	:param keep_prob: TF Placeholder for the dropout keep probability
	:param input_image: TF Placeholder for the image placeholder
	"""
	# Make folder for current run
	output_dir = os.path.join(runs_dir, str(time.time()))
	if os.path.exists(output_dir):
		shutil.rmtree(output_dir)
	os.makedirs(output_dir)

	# Run NN on test images and save them to HD
	print('Training Finished. Saving test images to: {}'.format(output_dir))
	image_outputs = gen_test_output(
		sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
	for name, image in image_outputs:
		imageio.imwrite(os.path.join(output_dir, name), image)
