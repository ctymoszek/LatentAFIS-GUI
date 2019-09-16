import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import utils
import model as modellib
import scipy.misc
import skimage.io

sys.path.append("./TensorVision")

#import TensorVision.utils as tv_utils
import utils as tv_utils

from hist_util import histeq
 

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


class ImportGraph():
    def __init__(self, model_dir):
        # create local graph and use it in the session
        self.model = load_model(model_dir)

    def run(self, original_image,gen_type = 'box', fuse_thres = 1):
        #feed_dict = {self.images_placeholder: img}
        #minutiae_cylinder = self.sess.run(self.minutiae_cylinder_placeholder, feed_dict=feed_dict)

        image_meta = original_image.shape
        original_image = scipy.misc.imresize(original_image.astype(float),
                                             (config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM), interp='bilinear')

        original_image = to_rgb(original_image)
        # Voting scheme: original image and equalized image are feed to give the result with variance of gray scale value is
        # from -10 to 10. The result is the region that have the 3/6 point value in area.

        # Initialize the cumulative mask as 0
        cumulative_mask = original_image[:, :, 0] * 0
        cumulative_box = original_image[:, :, 0] * 0

        # """
        # Normalize mean to 128:
        normal_image = np.uint8(original_image + 128 - np.mean(original_image))
        # original_image = normal_image

        # original_image has 3 channels as the same
        equalized_img_gray, _, _, _ = histeq(original_image[:, :, 0])
        # Processing Histogram equalized image
        equalized_img_rgb = to_rgb(equalized_img_gray)

        # Revert original_image
        # original_image = revert(original_image)

        cumulative_mask += calculate_cumulative_mask(self.model, normal_image, -10, 11)
        cumulative_mask += calculate_cumulative_mask(self.model, equalized_img_rgb, -10, 11)
        cumulative_mask += calculate_cumulative_mask(self.model, original_image, -10, 11)

        THRESHOLD_CUMULATIVE_MASK = fuse_thres

        thres_compare = min(THRESHOLD_CUMULATIVE_MASK, np.max(np.max(cumulative_mask)))

        # Bitwise compare to get the mask that each pixel has value of THRESHOLD_CUMULATIVE_MASK
        cumulative_mask = (cumulative_mask >= thres_compare) * 1

        # Bounding box wrap the mask
        pad = 0
        # Check if having mask return; if no mask, then box is the full image
        if np.sum(np.sum(cumulative_mask)) > 0:
            bbox = utils.extract_bboxes(to_one_depth(cumulative_mask))
            cumulative_box[max(0, bbox[0, 0] - pad):min(config.IMAGE_MAX_DIM - 1, bbox[0, 2] + pad),
            max(0, bbox[0, 1] - pad):min(config.IMAGE_MAX_DIM - 1, bbox[0, 3] + pad)] = 1
        else:
            cumulative_box[:, :] = 1
            cumulative_mask[:, :] = 1

        # Get the threshold size
        img_max_size = max(image_meta[1], image_meta[0])

        # Resize mask as the same size as img
        resized_cumulative_box = scipy.misc.imresize(cumulative_box.astype(float), (img_max_size, img_max_size),
                                                     interp='bilinear')
        resized_cumulative_box = resized_cumulative_box > 0

        resized_cumulative_mask = scipy.misc.imresize(cumulative_mask.astype(float), (img_max_size, img_max_size),
                                                      interp='bilinear')
        resized_cumulative_mask = resized_cumulative_mask > 0

        current_size = max(config.IMAGE_MAX_DIM, img_max_size)

        pad_width = (current_size - image_meta[0]) / 2
        pad_height = (current_size - image_meta[1]) / 2

        if pad_width >= 0 or pad_height >= 0:
            if gen_type == 'box':
                # SAVE_MASKED_IMG_DIR_BOX = CHANGE_PATH + '_box/'
                # if not os.path.exists(SAVE_MASKED_IMG_DIR_BOX):
                #    os.makedirs(SAVE_MASKED_IMG_DIR_BOX)
                # io.imsave(SAVE_MASKED_IMG_DIR_BOX + '%d_masked.bmp'%(image_id+1), resized_cumulative_box[pad_width:current_size - pad_width, pad_height:current_size - pad_height]*255)
                return resized_cumulative_box[pad_width:current_size - pad_width,
                       pad_height:current_size - pad_height] * 255, cumulative_mask * 255
            if gen_type == 'bmp':
                # SAVE_MASKED_IMG_DIR_BMP = CHANGE_PATH + '_bmp/'
                # if not os.path.exists(SAVE_MASKED_IMG_DIR_BMP):
                #    os.makedirs(SAVE_MASKED_IMG_DIR_BMP)
                # io.imsave(SAVE_MASKED_IMG_DIR_BMP + '%d_masked.bmp'%(image_id+1), resized_cumulative_mask[pad_width:current_size - pad_width, pad_height:current_size - pad_height]*255)
                return resized_cumulative_mask[pad_width:current_size - pad_width,
                       pad_height:current_size - pad_height] * 255, cumulative_mask * 255


from config import Config
class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 finger class

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    #For NISTSD27: 800, WVU: 1024 
    #IMAGE_MAX_DIM = 1024
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 500

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 15
    
    DETECTION_MIN_CONFIDENCE = 0
    
config = ShapesConfig()
#config.display()

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


class ShapesDataset(utils.Dataset):

    def load_shapes(self, DATA_DIR, height, width):
        """Create queue to load images into queue
        height, width: the size of the images.
        """
        # Add classes: prepare for general
        self.add_class("shapes", 1, "fingerprint")
        
        # Pair: (Image, Mask)
        # You can read all images from the path
        
        # Read Image from path
        name_images = []
        for root, subdirs, files in os.walk(DATA_DIR + '/image/'):
            for file in files:
                if os.path.splitext(file)[1].lower() in ('.bmp', '.jpg', '.jpeg'):
                    name_images.append(os.path.join(root, file))
        # Read Mask from path
        name_masks = []
        for root, subdirs, files in os.walk(DATA_DIR + '/Mask/'):
            for file in files:
                if os.path.splitext(file)[1].lower() in ('.bmp', '.jpg', '.jpeg'):
                    name_masks.append(os.path.join(root, file))
                    
        print("Total %d raw images" %len(name_images))
        print("Total %d raw masks" %len(name_masks))
        name_images = sorted(name_images)
        name_masks = sorted(name_masks)
        
        for i in range(0, len(name_images)):
            self.add_image("shapes", image_id=i, path=name_images[i],
                       width=width, height=height)
            self.add_mask("shapes", image_id=i, path=name_masks[i],
                       width=width, height=height)
        

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])

        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        return image

    def load_mask_image(self, mask_id):
        """Load the specified original mask image
        """
        
        # Load image
        image = skimage.io.imread(self.mask_info[mask_id]['path'])

        return image
    
    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

            
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        
        # Hardcoding -> TODO: need to change
        shapes = ['fingerprint']
        count = len(shapes)
        
        mask_o = skimage.io.imread(self.mask_info[image_id]['path'])/255

        mask = np.zeros([mask_o.shape[0], mask_o.shape[1], count], dtype=np.uint8)
        for i in range(0, count):
            mask[:, :, i] = mask_o        
        
        # Map class names to class IDs.
        class_ids = np.array([1])
        return mask, class_ids.astype(np.int32)


def imhist(im):
  # calculates normalized histogram of an image
	m, n = im.shape
	h = [0.0] * 256
	for i in range(m):
		for j in range(n):
			h[im[i, j]]+=1
	return np.array(h)/(m*n)

def cumsum(h):
	# finds cumulative sum of a numpy array, list
	return [sum(h[:i+1]) for i in range(len(h))]

def histeq(im):
	#calculate Histogram
	h = imhist(im)
	cdf = np.array(cumsum(h)) #cumulative distribution function
	sk = np.uint8(255 * cdf) #finding transfer function values
	s1, s2 = im.shape
	Y = np.zeros_like(im)
	# applying transfered values for each pixels
	for i in range(0, s1):
		for j in range(0, s2):
			Y[i, j] = sk[im[i, j]]
	H = imhist(Y)
	#return transformed image, original and new istogram, 
	# and transform function
	return Y , h, H, sk


class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    RPN_NMS_THRESHOLD = 0.7  # This is important because of threshold:
    DETECTION_MIN_CONFIDENCE = 0.3 # 0.5 is best Seems a little
    DETECTION_NMS_THRESHOLD = 0 # Seems nothing
    RPN_ANCHOR_STRIDE = 1
    RPN_ANCHOR_RATIOS = [0.25, 0.5, 0.75]
    #MEAN_PIXEL = [128, 128, 128]
    #ROI_POSITIVE_RATIO = 0.5 # seems nothing
    #RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    
    #RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)
    #ROI_POSITIVE_RATIO = 0.1
    #IMAGE_MAX_DIM = 800
    #IMAGE_MIN_DIM = 128

inference_config = InferenceConfig()
#inference_config.display()


def IoU(predict, groundtruth):
    intersection = predict & groundtruth
    union = predict | groundtruth
    iou1 = sum(sum(1.0*intersection))/sum(sum(1.0*union))

    predict = 1 - predict
    groundtruth = 1 - groundtruth
    
    intersection = predict & groundtruth
    union = predict | groundtruth
    iou2 = sum(sum(1.0*intersection))/sum(sum(1.0*union))
    return (iou1+iou2)/2

def to_rgb(im):
    # I think this will be slow
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret

def revert(image):
    image[:,:,0] = 255-image[:,:,0]
    image[:,:,1] = 255-image[:,:,1]
    image[:,:,2] = 255-image[:,:,2]  
    return image

def to_one_depth(im):
    # I think this will be slow
    w, h = im.shape
    ret = np.empty((w, h, 1), dtype=np.uint8)
    ret[:, :, 0] = im
    return ret

def calculate_cumulative_mask(model, base_image, begin_thres = -10, end_thres = 10, step = 10):
    # base_image has shape (w, h, 3)
    cumulative_mask = base_image[:,:,0]*0
    
    for thres in range(begin_thres, end_thres+1, step): # -10, 0, 10 is best for after hist equal
        image = base_image + thres
    
        # Run object detection
        results = model.detect([image], verbose=0, threshold_mask = 0.7)
        r = results[0]
        
        if r['masks'].shape[0] > 0:
            cumulative_mask += r['masks'][:,:,0]
        
    return cumulative_mask



#print("Images: {}\nClasses: {}".format(len(dataset_val.image_ids), dataset_val.class_names))

#MODEL_DIR =  "/research/prip-nguye590/Segmentation/Mask_RCNN/logs/shapes20171215T0924/mask_rcnn_shapes_0305.h5"
#MODEL_DIR =  "/research/prip-nguye590/Segmentation/Mask_RCNN/logs/shapes20171224T1316/mask_rcnn_shapes_0450.h5"
#MODEL_DIR =  "/research/prip-nguye590/Segmentation/Mask_RCNN/logs/shapes20171228T1329/mask_rcnn_shapes_0033.h5"
#MODEL_DIR =  "/research/prip-nguye590/Segmentation/Mask_RCNN/logs/shapes20171228T1329/mask_rcnn_shapes_0140.h5"


CHANGE_PATH = '/research/prip-nguye590/Segmentation/Mask_RCNN/Masked_image/send_Kai_v2'

def load_model(MODEL_DIR):
    # Create model in inference mode
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                  config=inference_config)

    weights_path = MODEL_DIR

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)
    return model
    

def generate_mask(model, original_image, DATA_DIR, gen_type = 'bmp', fuse_thres = 1):
    image_meta = original_image.shape
    original_image = scipy.misc.imresize(original_image.astype(float), (config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM), interp='bilinear')

    original_image = to_rgb(original_image)
    # Voting scheme: original image and equalized image are feed to give the result with variance of gray scale value is
    # from -10 to 10. The result is the region that have the 3/6 point value in area.

    # Initialize the cumulative mask as 0
    cumulative_mask = original_image[:,:,0]*0
    cumulative_box = original_image[:,:,0]*0

    #"""
    # Normalize mean to 128:
    normal_image = np.uint8(original_image + 128 - np.mean(original_image))
    #original_image = normal_image

    # original_image has 3 channels as the same
    equalized_img_gray, _, _, _ = histeq(original_image[:,:,0])
    # Processing Histogram equalized image
    equalized_img_rgb = to_rgb(equalized_img_gray)

    # Revert original_image
    #original_image = revert(original_image)

    cumulative_mask += calculate_cumulative_mask(model, normal_image, -10, 11)
    cumulative_mask += calculate_cumulative_mask(model, equalized_img_rgb, -10, 11)
    cumulative_mask += calculate_cumulative_mask(model, original_image, -10, 11)

    THRESHOLD_CUMULATIVE_MASK = fuse_thres

    thres_compare = min(THRESHOLD_CUMULATIVE_MASK,np.max(np.max(cumulative_mask)))

    # Bitwise compare to get the mask that each pixel has value of THRESHOLD_CUMULATIVE_MASK
    cumulative_mask = (cumulative_mask >= thres_compare)*1

    # Bounding box wrap the mask
    pad = 0
    # Check if having mask return; if no mask, then box is the full image
    if np.sum(np.sum(cumulative_mask))>0:
        bbox = utils.extract_bboxes(to_one_depth(cumulative_mask))
        cumulative_box[max(0,bbox[0,0]-pad):min(config.IMAGE_MAX_DIM-1,bbox[0,2]+pad), max(0,bbox[0,1]-pad):min(config.IMAGE_MAX_DIM-1,bbox[0,3]+pad)] = 1
    else:
        cumulative_box[:,:] = 1
        cumulative_mask[:, :] = 1


    # Get the threshold size
    img_max_size = max(image_meta[1], image_meta[0])




    # Resize mask as the same size as img
    resized_cumulative_box = scipy.misc.imresize(cumulative_box.astype(float), (img_max_size, img_max_size), interp='bilinear')
    resized_cumulative_box = resized_cumulative_box > 0

    resized_cumulative_mask = scipy.misc.imresize(cumulative_mask.astype(float), (img_max_size, img_max_size), interp='bilinear')
    resized_cumulative_mask = resized_cumulative_mask > 0

    current_size = max(config.IMAGE_MAX_DIM, img_max_size)

    pad_width = (current_size - image_meta[0])/2
    pad_height = (current_size - image_meta[1])/2




    if pad_width >= 0 or pad_height >= 0:
        if gen_type == 'box':
            #SAVE_MASKED_IMG_DIR_BOX = CHANGE_PATH + '_box/'
            #if not os.path.exists(SAVE_MASKED_IMG_DIR_BOX):
            #    os.makedirs(SAVE_MASKED_IMG_DIR_BOX)
            #io.imsave(SAVE_MASKED_IMG_DIR_BOX + '%d_masked.bmp'%(image_id+1), resized_cumulative_box[pad_width:current_size - pad_width, pad_height:current_size - pad_height]*255)
            return resized_cumulative_box[pad_width:current_size - pad_width, pad_height:current_size - pad_height]*255, cumulative_mask*255
        if gen_type == 'bmp':
            #SAVE_MASKED_IMG_DIR_BMP = CHANGE_PATH + '_bmp/'
            #if not os.path.exists(SAVE_MASKED_IMG_DIR_BMP):
            #    os.makedirs(SAVE_MASKED_IMG_DIR_BMP)
            #io.imsave(SAVE_MASKED_IMG_DIR_BMP + '%d_masked.bmp'%(image_id+1), resized_cumulative_mask[pad_width:current_size - pad_width, pad_height:current_size - pad_height]*255)
            return resized_cumulative_mask[pad_width:current_size - pad_width, pad_height:current_size - pad_height]*255, cumulative_mask*255

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model_dir = '/home/kaicao/Research/Mask_framework/15_12_0140.h5'
    model = ImportGraph(model_dir)
    img = plt.imread('/home/kaicao/Dropbox/Research/Data/Latent/NISTSD27/image/001.bmp')
    mask,_ = model.run(img, gen_type='box', fuse_thres=1)
    print mask.shape
    plt.set_cmap('gray')
    plt.imshow(mask)
    plt.show()