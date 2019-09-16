import glob
import os.path
import pdb
from shutil import copyfile

#pathname = '/media/kaicao/Exchange/Descriptor/TrainingMinutiaePatch_Enh_1131/'
pathname = '//media/kaicao/data2/AutomatedLatentRecognition/Data/OF/images/'
imagefiles = glob.glob(pathname+'*.jpeg')
imagefiles.sort()
fname = os.path.basename(imagefiles[-1])

N = 0
for file in imagefiles:
	basename = os.path.basename(file)
	n = int(basename.split('_')[0])
	#print n
	
	if n>N:
		N = n

print('total number of fingers: %d'%N)
#t_pathname = '/media/kaicao/Exchange/Descriptor/TrainingMinutiaePatch_Enh_1131_subfolders/'
#t_pathname = '/scratch/LatentAFIS/Data/Descriptor/TrainingMinutiaePatch_Enh_1131_subfolders/'
t_pathname = '/media/kaicao/data2/AutomatedLatentRecognition/Data/OF/images_subfolders/'
if not os.path.exists(t_pathname):
        os.makedirs(t_pathname)

for i in range(N): 
	directory = t_pathname + str(i+1) + '/'
	if not os.path.exists(directory):
		os.makedirs(directory)

for file in imagefiles:
	basename = os.path.basename(file)
	finger = basename.split('_')[0]
	#print n
	
	directory = t_pathname + finger + '/'
	copyfile(file,directory+basename)
print N	
print fname
print len(imagefiles)
