# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pdb
import faiss
from time import time
import glob
import sys
import scipy.io as sio
import os
import pickle
import timeit
import time
import itertools
from numba import jit
sys.path.append('../')
import template

@jit
def get_all_faetures_MSP(template_path=None, N=1000):
	if N<1:
		return None
	minutiae = []
	finger_ID = []
	des = []
	for i in range(1,N+1):
		print i
		fname = template_path +str(i) + '.dat'
		rolled_template = template.Bin2Template_Byte_TF(fname, isLatent=True)
		minu_set_num = len(rolled_template.minu_template)
		if minu_set_num == 0:
			continue
		minu_num = len(rolled_template.minu_template[0].minutiae)
		if minu_num == 0:
			continue
		tmp_minu = rolled_template.minu_template[0].minutiae
		minutiae.extend(tmp_minu)
		finger_ID.extend(np.zeros(minu_num, ) + i)
		des.extend(rolled_template.minu_template[0].des[0])
	des = np.asarray(des)
	minutiae = np.asarray(minutiae)
	finger_ID = np.asarray(finger_ID)
	return des, finger_ID, minutiae

#@jit
def get_all_faetures_MSP2(template_path=None, N=1000):
	if N<1:
		return None
	minutiae = []
	finger_ID = []
	des = []
	# minu_num_total = 0
	# des_num = 0
	# for i in range(1,N):
    #
	# 	print i
	# 	fname = template_path +str(i) + '.dat'
	# 	rolled_template = template.Bin2Template_Byte_TF(fname, isLatent=True)
	# 	if i==1:
	# 		des_num = rolled_template.minu_template[0].des[0]
    #
	# 	minu_set_num = len(rolled_template.minu_template)
	# 	if minu_set_num == 0:
	# 		continue
	# 	minu_num = len(rolled_template.minu_template[0].minutiae)
	# 	minu_num_total += minu_num
	# des = np.zeros((minu_num_total,des_num))
	# finger_ID = np.zeros((minu_num_total,))
	# minutiae = np.zeros((minu_num_total,4))
    #
	# start = 0

	for i in range(1, N+1):
		print i
		fname = template_path + str(i) + '.dat'
		rolled_template = template.Bin2Template_Byte_TF(fname, isLatent=True)
		if i == 1:
			des_num = rolled_template.minu_template[0].des[0]

		minu_set_num = len(rolled_template.minu_template)
		if minu_set_num == 0:
			continue
		minu_num = len(rolled_template.minu_template[0].minutiae)
		if minu_num == 0:
			continue
		tmp_minu = rolled_template.minu_template[0].minutiae
		minutiae.append(tmp_minu)
		finger_ID.extend(np.zeros(minu_num, ) + i)
		des.extend(rolled_template.minu_template[0].des[0])
	#des = list(itertools.chain.from_iterable(des))
	des = np.asarray(des)
	minutiae = np.asarray(minutiae)
	finger_ID = np.asarray(finger_ID)
	return des, finger_ID, minutiae
def create_index(template_path,N=10000,index_file=None, gallery_file =None):
	
  	
  	#gallery_file ='/media/kaicao/data2/AutomatedLatentRecognition/Results/template/MSP_100K.mat'
  	if os.path.exists(gallery_file):
  		D_gallery = sio.loadmat(gallery_file)
  	else:
		des, finger_ID, minutiae = get_all_faetures_MSP2(template_path = template_path,N=N)
		D_gallery = {}
		D_gallery['des'] = des
		D_gallery['finger_ID'] = finger_ID
		D_gallery['minutiae'] = minutiae
		sio.savemat(gallery_file,D_gallery)
	
	# query_file = '/media/kaicao/data2/AutomatedLatentRecognition/Results/template/NISTSD14_S.mat'
 #  	if os.path.exists(query_file):
 #  		D_query = sio.loadmat(query_file)
 #  	else:
	# 	des, finger_ID, minutiae = get_all_faetures(template_path = template_path,prefix = 'S')
	# 	D_query = {}
	# 	D_query['des'] = des
	# 	D_query['finger_ID'] = finger_ID
	# 	D_query['minutiae'] = minutiae
	# 	sio.savemat(query_file,D_query)
	
	finger_ID = D_gallery['finger_ID']
	minutiae = D_gallery['minutiae'] 
	des = D_gallery['des'].copy().astype('float32')
	#des = des[:1280,:32]
	print des.shape
	del D_gallery 
	dim = des.shape[1]             # feature dimension

	nlist = 50
	m = 16
	quantizer = faiss.IndexFlatL2(dim)  # this remains the same
	index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)
	#pdb.set_trace()
	index.train(des)
	index.add(des)
	if index_file is not None:
		faiss.write_index(index, index_file)

def search(query_path,index_file,gallery_file):
	index = faiss.read_index(index_file)
	gallery = sio.loadmat(gallery_file)
	gallery_finger_ID = gallery['finger_ID'].astype(np.int)

	gallery_size = np.max(gallery_finger_ID)+1
	template_list = glob.glob(query_path+'*.dat')
	template_list.sort()
	minutiae = []
	des = []
	#[14 8 16 10]
	candi_size = 100
	query_num = len(template_list)
	rank = np.zeros((query_num,)) + query_num

	t = 0
	#start = timeit.timeit()
	start = time.time()
	for i, file in enumerate(template_list):
		#if i<2:
		#	continue

		# if i>10000:
		# 	break
		rolled_template = template.Bin2Template_Byte_TF(file,isLatent=True)
		minu_set_num = len(rolled_template.minu_template)
		if minu_set_num == 0:
			continue
		des = []
		for n in range(2,3): #minu_set_num):
			minu_num = len(rolled_template.minu_template[n].minutiae)
			if minu_num == 0:
				continue
			tmp_des = rolled_template.minu_template[n].des[0]
			des.extend(tmp_des)

		# minutiae = rolled_template.minu_template[1].minutiae
		# # minutiae = i
		# des = rolled_template.minu_template[1].des[0]
		des = np.asarray(des)
		dist, I = index.search(des, candi_size)     # search
		simi = 1 - dist
		score = np.zeros((gallery_size,))
		#pdb.set_trace()
		for k in range(minu_num):
			#minu_score = np.zeros((gallery_size,))
			for j in range(candi_size):
				if simi[k,j]<0:
					continue
				#pdb.set_trace()
				# if I[k,j]<30:
				# 	print I[k,j]
				ID = gallery_finger_ID[0,I[k,j]]-1
				# if ID == 0:
				# 	print ID
				# if minu_score[ID]<simi[k,j]:
				# 	minu_score[ID] = simi[k,j]
				score[ID] +=  simi[k,j]
			# score += minu_score
		order = np.argsort(score)[::-1]
		tmp = np.argwhere(order==i)
		rank[i] = tmp[0,0]+1
	#end = timeit.timeit()
	end = time.time()
	t = end - start
	print t
	print len(np.argwhere(rank <= 20))*1.0/query_num
	return rank
	#pdb.set_trace()
	#print 'average search time = %f' % t/query_num
if __name__ == '__main__':

	os.environ['CUDA_VISIBLE_DEVICES'] = '0'

	#template_path = '/media/kaicao/Seagate Backup Plus Drive/Research/AutomatedLatentRecognition/template/MSP/version_2/'
	template_path = '/media/kaicao/data2/AutomatedLatentRecognition/Results/template/MSP/version_2/'
	index_file = "/media/kaicao/data2/AutomatedLatentRecognition/Results/template/index_gallery_MSP_100K.index"
	gallery_file = '/media/kaicao/data2/AutomatedLatentRecognition/Results/template/MSP_100K.mat'
	# create index for 100K MSP background
	start = timeit.default_timer()
	#des, finger_ID, minutiae = get_all_faetures_MSP2(template_path=template_path, N=100000)


	create_index(template_path,N=100000, index_file = index_file,gallery_file = gallery_file)
	end = timeit.default_timer()
	print end-start

	# conduct retrieval experiments on NISTSD27
	latent_template_path = '/media/kaicao/data2/AutomatedLatentRecognition/Results/template/latent/NISTSD27_AEM2/'
	rank = search(latent_template_path,index_file,gallery_file)
	print len(np.argwhere(rank <= 100)) * 1.0 / len(rank)
	#print rank
    #sio.savemat('/media/kaicao/data2/AutomatedLatentRecognition/Results/template/index_gallery_NISTSD14.mat',index)
	#pdb.set_trace()

# MSP = np.load('../feature/MSP_CrossEntropy.npy')
# NISTSD4 = np.load('../feature/NISTSD4.npy')
# Gallery = np.concatenate((NISTSD4[:2000,:],MSP),axis=0)
# #Gallery = NISTSD4[:2000,:]
# Query = NISTSD4[2000:,:].astype('float32')
# Gallery = Gallery.astype('float32')

# d = Gallery.shape[1]             # feature dimension

# #pdb.set_trace()

# nlist = 100
# m = 16
# k = 252                          # top 0.1%
# quantizer = faiss.IndexFlatL2(d)  # this remains the same
# index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
#                                   # 8 specifies that each sub-vector is encoded as 8 bits
# index.train(Gallery)
# index.add(Gallery)
# #D, I = index.search(xb[:5], k) # sanity check
# #print I
# #print D
# start_time = time()
# index.nprobe = 20              # make comparable with experiment above
# D, I = index.search(Query, k)     # search
# end_time = time()
# print('search time for %d queres: %f' % (Query.shape[0],end_time-start_time))
# correct_num = 0
# for i in range(Query.shape[0]):
#     if len(np.where(I[i]==i)[0]) ==1:
# 	correct_num += 1

# print correct_num

# #pdb.set_trace()
