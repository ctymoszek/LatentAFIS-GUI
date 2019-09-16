img_path = '/research/prip-kaicao/Data/Rolled/NIST4/Image_Aligned/';
target_path = '/research/prip-kaicao/Data/Rolled/NIST4/Image_Aligned_0.65/';
if ~isdir(target_path)
    mkdir(target_path)
end
imglist = dir([img_path '*.jpeg']);

for i=1:length(imglist)
    i
    img = imread([img_path imglist(i).name]);
    img = imresize(img,0.65);
    imwrite(img,[target_path imglist(i).name]);
end