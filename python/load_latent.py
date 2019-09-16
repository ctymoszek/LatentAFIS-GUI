import sys
import os
from shutil import rmtree
from matching.Matching import RunMatcher
from GUI.feature_images import SaveFeatureImages, SaveCorrespondenceImage

latent_img_path = sys.argv[1]
latent_data_path = sys.argv[2]
head, tail = os.path.split(unicode(latent_img_path))
dir = os.path.dirname(__file__)
# latent_img_path = os.path.join(dir, "Data/Latent/" + latent_img_path)
# latent_data_path = os.path.join(dir, "Data/Latent/" + latent_data_path)

#get feature images
feature_imgs_path = os.path.join(dir, "Data/current_latent_data/")
if os.path.exists(feature_imgs_path):
    rmtree(feature_imgs_path)
os.makedirs(feature_imgs_path)
SaveFeatureImages(latent_img_path, latent_data_path, feature_imgs_path)
