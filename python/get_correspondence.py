import os
import sys
from skimage import io
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from matplotlib.patches import ConnectionPatch

plt.switch_backend('Agg')

latent_img_path = sys.argv[1]
rolled_img_path = sys.argv[2]

feature_imgs_path = "/home/cori/research/LatentAFISV2/Matching_20181204/scores/corr/"

corr_img_path = feature_imgs_path + os.path.splitext(os.path.split(latent_img_path)[1])[0] + "_" + os.path.splitext(os.path.split(rolled_img_path)[1])[0] + "_2.jpg"

corr_dat_path = feature_imgs_path + os.path.splitext(os.path.split(latent_img_path)[1])[0] + "_" + os.path.splitext(os.path.split(rolled_img_path)[1])[0] + "_2.csv"

# print(corr_img_path)
# print(corr_dat_path)

pairs = []
with open(corr_dat_path) as f:
    reader = csv.reader(f)
    for row in reader:
        pairs.append([int(x) for x in row])

latent_img = io.imread(latent_img_path)
rolled_img = io.imread(rolled_img_path)

fig, (ax1, ax2) = plt.subplots(1,2)
ax1.imshow(latent_img, interpolation='nearest', cmap=plt.cm.gray)
ax2.imshow(rolled_img, interpolation='nearest', cmap=plt.cm.gray)
ax1.axis('off')
ax2.axis('off')
plt.gca().set_axis_off()
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(NullLocator())
plt.gca().yaxis.set_major_locator(NullLocator())

artists = []
for i in range(len(pairs)):
    l_x = pairs[i][0]
    l_y = pairs[i][1]
    r_x = pairs[i][2]
    r_y = pairs[i][3]
    # print("Connecting " + "(" + str(l_x) + ", " + str(l_y) + ") with (" + str(r_x) + ", " + str(r_y) + ")")
    con = ConnectionPatch(xyA=(r_x, r_y), xyB=(l_x, l_y), coordsA='data', coordsB='data', axesA=ax2, axesB=ax1, color='red', alpha=0.7)
    ax2.add_artist(con)
    artists.append(con)

# print(latent_img.shape)
# print(rolled_img.shape)
# ax1.set_xlim(latent_img.shape[1])
# ax1.set_ylim(latent_img.shape[0])
# ax2.set_xlim(rolled_img.shape[1])
# ax2.set_ylim(rolled_img.shape[0])

fig.savefig(corr_img_path, dpi=600, bbox_inches='tight', pad_inches=0.0)

print(len(pairs))