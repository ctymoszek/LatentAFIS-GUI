CUDA_VISIBLE_DEVICES=6,python src/feature_extraction.py \
--data_dir ~/prip-kaicao/Data/Rolled/NIST4/Image_Aligned/ \
--model_dir /research/prip-kaicao/models/indexing_cross_entropy_image_size_512/20170720-103325/  \
--batch_size 32 \
--fname feature/NISTSD4_512_top_middle.npy \
--cx 106 \
--cy 16 

#CUDA_VISIBLE_DEVICES=6, python src/feature_extraction.py \
#--data_dir ~/prip-kaicao/Data/Rolled/NIST4/Image_Aligned/ \
#--model_dir /research/prip-kaicao/models/indexing_cross_entropy_image_size_512/20170720-103325/  \
#--batch_size 32 \
#--fname feature/NISTSD4_512_top_right.npy \
#--cx 196 \
#--cy 16 

CUDA_VISIBLE_DEVICES=6, python src/feature_extraction.py \
--data_dir ~/prip-kaicao/Data/Rolled/NIST4/Image_Aligned/ \
--model_dir /research/prip-kaicao/models/indexing_cross_entropy_image_size_512/20170720-103325/  \
--batch_size 32 \
--fname feature/NISTSD4_512_middle_left.npy \
--cx 16 \
--cy 106 

CUDA_VISIBLE_DEVICES=6, python src/feature_extraction.py \
--data_dir ~/prip-kaicao/Data/Rolled/NIST4/Image_Aligned/ \
--model_dir /research/prip-kaicao/models/indexing_cross_entropy_image_size_512/20170720-103325/  \
--batch_size 32 \
--fname feature/NISTSD4_512_middle_middle.npy \
--cx 106 \
--cy 106 

CUDA_VISIBLE_DEVICES=6, python src/feature_extraction.py \
--data_dir ~/prip-kaicao/Data/Rolled/NIST4/Image_Aligned/ \
--model_dir /research/prip-kaicao/models/indexing_cross_entropy_image_size_512/20170720-103325/  \
--batch_size 32 \
--fname feature/NISTSD4_512_middle_right.npy \
--cx 196 \
--cy 106

CUDA_VISIBLE_DEVICES=6, python src/feature_extraction.py \
--data_dir ~/prip-kaicao/Data/Rolled/NIST4/Image_Aligned/ \
--model_dir /research/prip-kaicao/models/indexing_cross_entropy_image_size_512/20170720-103325/  \
--batch_size 32 \
--fname feature/NISTSD4_512_bottom_left.npy \
--cx 16 \
--cy 196

CUDA_VISIBLE_DEVICES=6, python src/feature_extraction.py \
--data_dir ~/prip-kaicao/Data/Rolled/NIST4/Image_Aligned/ \
--model_dir /research/prip-kaicao/models/indexing_cross_entropy_image_size_512/20170720-103325/  \
--batch_size 32 \
--fname feature/NISTSD4_512_bottom_middle.npy \
--cx 106 \
--cy 196

CUDA_VISIBLE_DEVICES=6, python src/feature_extraction.py \
--data_dir ~/prip-kaicao/Data/Rolled/NIST4/Image_Aligned/ \
--model_dir /research/prip-kaicao/models/indexing_cross_entropy_image_size_512/20170720-103325/  \
--batch_size 32 \
--fname feature/NISTSD4_512_bottom_right.npy \
--cx 196 \
--cy 196

