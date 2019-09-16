CUDA_VISIBLE_DEVICES=7, python src/feature_extraction.py \
--data_dir ~/prip-kaicao/Data/Rolled/NIST4/Image_Aligned_0.65/ \
--model_dir /research/prip-kaicao/models/indexing_triplet/20170718-075624/ \
--batch_size 32 \
--fname feature/NISTSD4_triplet.npy

