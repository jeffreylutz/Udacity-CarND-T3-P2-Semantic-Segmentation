import cv2
import os.path
import scipy.misc


def make_video(images, outvid=None, format="XVID", fps=5, size=None, is_color=True):
    """
    Create a video from a list of images.

    @param      images      list of images to use in the video
    @param      outvid      output video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    fourcc = VideoWriter_fourcc(*format)
    vidWriter = None
    i = 1
    for image in images:
        i = i+1
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vidWriter is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vidWriter = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vidWriter.write(img)
    vidWriter.release()
    return vidWriter


#image = scipy.misc.imresize(scipy.misc.imread('1_before.png'), image_shape)
image = scipy.misc.imread('1_before.png')
image_filename = '1_before.png'
image = cv2.imread(image_filename)
images = [image_filename,image_filename,image_filename,image_filename,image_filename,image_filename,image_filename,image_filename,image_filename,image_filename]
# works.  Note the AVI extension and type: MJPG and the frame size being flipped w,h->h,w
make_video(images, 'myvid.avi', "MJPG", 1, None, True)

print("Finished writing...")