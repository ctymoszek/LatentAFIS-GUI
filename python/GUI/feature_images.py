import os
from skimage import io
from matching.Bin2Template_Byte_TF import Bin2Template_Byte_TF
import show
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from matplotlib.patches import ConnectionPatch

def SaveFeatureImages(img_file, data_file, feature_img_path):
    img = io.imread(img_file)
    name = os.path.basename(img_file)
    head, tail = os.path.splitext(name)
    h, w = img.shape
    block = True

    template = Bin2Template_Byte_TF(data_file, isLatent=True)

    mnt = template.minu_template[0].minutiae
    fname = os.path.join(unicode(feature_img_path), head + "_minu1.jpg")
    show.show_minutiae(img, mnt, block=block, fname=fname)

    mnt = template.minu_template[1].minutiae
    fname = os.path.join(unicode(feature_img_path), head + "_minu2.jpg")
    show.show_minutiae(img, mnt, block=block, fname=fname)

    mask = template.minu_template[0].mask
    fname = os.path.join(unicode(feature_img_path), head + "_ROI.jpg")
    show.show_mask(mask, img, fname=fname, block=block)

    OF = template.minu_template[0].oimg
    fname = os.path.join(unicode(feature_img_path), head + "_OF.jpg")
    show.show_orientation_field(img, OF, mask=mask, fname=fname)

    return

def SaveCorrespondenceImage(latent_data_path, latent_img_path, rolled_img_path,
                            rolled_data_path, corr_img_path, corr):
    latent_img = io.imread(latent_img_path)
    rolled_img = io.imread(rolled_img_path)

    latent_template = Bin2Template_Byte_TF(latent_data_path, isLatent=True)
    rolled_template = Bin2Template_Byte_TF(rolled_data_path, isLatent=False)

    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(latent_img, interpolation='nearest', cmap=plt.cm.gray)
    ax2.imshow(rolled_img, interpolation='nearest', cmap=plt.cm.gray)
    ax1.axis('off')
    ax2.axis('off')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())

    for i in range(len(corr[0])):
        l_x = latent_template.minu_template[0].minutiae[corr[0][i]][0]
        l_y = latent_template.minu_template[0].minutiae[corr[0][i]][1]
        r_x = rolled_template.minu_template[0].minutiae[corr[1][i]][0]
        r_y = rolled_template.minu_template[0].minutiae[corr[1][i]][1]
        con = ConnectionPatch(xyA=(r_x,r_y),xyB=(l_x,l_y),coordsA='data',coordsB='data',axesA=ax2,axesB=ax1,color='red')
        ax2.add_artist(con)

    ax1.set_xlim(0,latent_template.minu_template[0].w)
    ax1.set_ylim(latent_template.minu_template[0].h,0)
    ax2.set_xlim(0,rolled_template.minu_template[0].w)
    ax2.set_ylim(rolled_template.minu_template[0].h,0)

    fig.savefig(corr_img_path, dpi = 600, bbox_inches='tight', pad_inches=0.0)

    return
