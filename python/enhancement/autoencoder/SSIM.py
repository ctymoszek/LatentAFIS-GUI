import tensorflow as tf
import numpy as np

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a different scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # list to tensor of dim D+1
    mssim = tf.pack(mssim, axis=0)
    mcs = tf.pack(mcs, axis=0)

    value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                            (mssim[level-1]**weight[level-1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value

if __name__ == '__main__':
    size = 11
    sigma = 1.5
    t = _tf_fspecial_gauss(size, sigma)
#
# import numpy as np
# import tensorflow as tf
# from skimage import data, img_as_float
#
# image = data.camera()
# img = img_as_float(image)
# rows, cols = img.shape
#
# noise = np.ones_like(img) * 0.2 * (img.max() - img.min())
# noise[np.random.random(size=noise.shape) > 0.5] *= -1
#
# img_noise = img + noise
#
# ## TF CALC START
# BATCH_SIZE = 1
# CHANNELS = 1
# image1 = tf.placeholder(tf.float32, shape=[rows, cols])
# image2 = tf.placeholder(tf.float32, shape=[rows, cols])
#
# def image_to_4d(image):
#     image = tf.expand_dims(image, 0)
#     image = tf.expand_dims(image, -1)
#     return image
#
# image4d_1 = image_to_4d(image1)
# image4d_2 = image_to_4d(image2)
#
# ssim_index = tf_ssim(image4d_1, image4d_2)
#
# msssim_index = tf_ms_ssim(image4d_1, image4d_2)
#
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#
#     tf_ssim_none = sess.run(ssim_index,
#                             feed_dict={image1: img, image2: img})
#     tf_ssim_noise = sess.run(ssim_index,
#                              feed_dict={image1: img, image2: img_noise})
#
#     tf_msssim_none = sess.run(msssim_index,
#                             feed_dict={image1: img, image2: img})
#     tf_msssim_noise = sess.run(msssim_index,
#                              feed_dict={image1: img, image2: img_noise})
# ###TF CALC END
#
# print('tf_ssim_none', tf_ssim_none)
# print('tf_ssim_noise', tf_ssim_noise)
# print('tf_msssim_none', tf_msssim_none)
# print('tf_msssim_noise', tf_msssim_noise)