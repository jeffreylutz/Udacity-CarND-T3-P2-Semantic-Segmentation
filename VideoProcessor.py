import sys
import tensorflow as tf
import numpy as np
import scipy.misc
from moviepy.editor import VideoFileClip


class VideoProcessor(object):
    """A VideoProcessor for Semantic Segmentation"""

    def __init__(self):
        print("Attempting to load model...")
        self.image_shape = (160, 576)

        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph('checkpoints/semantic-segmentation-model.meta')
        self.saver.restore(self.sess, tf.train.latest_checkpoint('checkpoints/.'))
        self.graph = tf.get_default_graph()
        self.logits = self.graph.get_tensor_by_name("logits:0")
        self.keep_prob = self.graph.get_tensor_by_name("keep_prob:0")
        self.image_input = self.graph.get_tensor_by_name('image_input:0')
        print("Loaded existing model.")

    def process_video(self, in_video_filename, out_video_filename):
        print("MOVIE IN:",in_video_filename)
        print("MOVIE OUT:",out_video_filename)
        vid = VideoFileClip(filename=in_video_filename, audio=False)
        clip1 = vid.subclip()
        white_clip = clip1.fl_image(self.process_image)
        white_clip.write_videofile(out_video_filename, audio=False)

    def process_image(self, image):
        """Processes video filename with semantic segmentation"""

        # Retain original image shape
        origShape = image.shape
        # Resize image to specified shape
        image = scipy.misc.imresize(image, self.image_shape)

        # Create image softmax (aka image mask)
        im_softmax = self.sess.run(
            [tf.nn.softmax(self.logits)]
            , {self.keep_prob: 1.0, self.image_input: [image]}
        )
        im_softmax = im_softmax[0][:, 1].reshape(self.image_shape[0],
                                                 self.image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(self.image_shape[0],
                                                  self.image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        # Convert numpy array to scipy image in order to make a copy
        out_image = scipy.misc.toimage(image)
        # Apply mask to copy of image (out_image)
        out_image.paste(mask, box=None, mask=mask)
        # convert image back to numpy array
        out_image = scipy.misc.fromimage(out_image)

        return out_image


# Get the total number of args passed
total = len(sys.argv)

if(total != 3):
    print("USAGE:  ",str(sys.argv[0])," <input_mp4> <output_mp4>")
    exit(1)

movie_in = str(sys.argv[1])
movie_out = str(sys.argv[2])
print(movie_in)


# image_shape = (160, 576)
# image = scipy.misc.imresize(scipy.misc.imread('1_before.png'), image_shape)

vid = VideoProcessor()
vid.process_video(movie_in, movie_out)
