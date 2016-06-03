import numpy as np
from skimage.transform import resize
import caffe
import matplotlib.pyplot as plt
from subprocess import call

plt.rcParams['image.cmap'] = 'gray'

PROTO_PATH = 'deploy_viz.prototxt'
MODEL_PATH = 'VGG_ILSVRC_16_layers.caffemodel'

def init():
	global net, transformer

	caffe.set_mode_cpu()
	net = caffe.Net(PROTO_PATH, MODEL_PATH, caffe.TEST)	

	input_shape = net.blobs['data'].data.shape
	transformer = caffe.io.Transformer({'data': input_shape})
	transformer.set_transpose('data', (2,0,1))
	transformer.set_mean('data', np.array([103.939, 116.779, 123.68]))
	transformer.set_raw_scale('data', 255)
	transformer.set_channel_swap('data', (2,1,0))

def hotmap(image):
	global net, transformer

	transformed_image = transformer.preprocess('data', image)
	net.blobs['data'].data[...] = transformed_image	

	output = net.forward()
	output_prob = output['prob']
	output_shape = output_prob.shape	
	input_shape = net.blobs['data'].data.shape

	idx = np.argmax(output_prob)	

	error = np.zeros(output_prob.shape)
	error[:, idx] = 1	

	diffs = net.backward(**{'prob':error})
	classvalue = np.max(diffs['data'][0], 0)
	classvalue = resize(classvalue, image.shape[:2], order=3)

	classvalue = np.abs(classvalue)

	max_v, min_v = np.max(classvalue), np.min(classvalue)
	classvalue = (classvalue - min_v) / (max_v - min_v)

	plt.imshow(classvalue)
	plt.savefig('hotmap.jpg')

	with open('hotmap', 'w') as f:
		shape = classvalue.shape
		for i in range(shape[0]):
			for j in range(shape[1]):
				f.write(str(classvalue[i,j])+'\n')

if __name__ == '__main__':
	init()

	while True:
		NAME = raw_input("Input image path:")
		if NAME == 'Q':
			break
		IMAGE_PATH = 'imgs/' + NAME
		image = caffe.io.load_image(IMAGE_PATH)
		hotmap(image)
		print 'Hot map generated!'

		call(["./GraphCut/build/GraphCut", IMAGE_PATH, 'hotmap'])
	# 	call(["mv", 'result.jpg', 'results/'+NAME])
	# 	call(["mv", 'hotmap.jpg', 'hotmaps/'+NAME])

	# from os import listdir
	# from os.path import isfile, join
	# files = [f for f in listdir('imgs') if isfile(join('imgs', f)) and not f.startswith('.')]
	# for NAME in files:
	# 	IMAGE_PATH = 'imgs/' + NAME
	# 	image = caffe.io.load_image(IMAGE_PATH)
	# 	hotmap(image)
	# 	print 'Hot map generated!'

	# 	call(["./GraphCut/build/GraphCut", IMAGE_PATH, 'hotmap'])
	# 	call(["mv", 'result.jpg', 'results/'+NAME])
	# 	call(["mv", 'hotmap.jpg', 'hotmaps/'+NAME])