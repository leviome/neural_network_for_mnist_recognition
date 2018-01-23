'''
@author : liaoliwei
'''
import tensorflow as tf
import numpy as np
import mnist_inference
import pre
import os
import mnist_train

def pic(path):
	with tf.Graph().as_default() as g:
		x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input') 
		pic_feed = {x:pre.pre(path)}
		y = mnist_inference.inference(x, None)
		variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
		variables_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)
		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
			print(ckpt)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				yy = sess.run(y, feed_dict=pic_feed)
				res = sess.run(tf.argmax(yy,1))
				print(res)
			else:
				print("No checkpoint file found!")
				return
def main(argv=None):
	path = input("input path:")
	pic(path)

if __name__=="__main__":
	main()
