%cnn_latent
run(fullfile(fileparts(mfilename('fullpath')), '../matlab/vl_setupnn.m')) ;
pathname = '../../Data/Latent/NISTSD27/image/';
maskpath = '../../Data/Latent/NISTSD27/maskNIST27/';
load('data/OFClassNo_128/model2/net-epoch-12_cpu.mat','net');
OF_net = net;
load('data/OFClassNo_128/OriCenter.mat','center');
OF_center = center;

% load('data/RFClassNo_32/model_RF_1/net-epoch-38_cpu.mat','net');
% RF_net = net;
% load('data/RFClassNo_32/FreCenter.mat','center');
% RF_center = center;

addpath('Common')
% reduce the resolution of center

[centernum,dim] = size(OF_center);

cos2Theta = cell(centernum,1); 
sin2Theta = cell(centernum,1);
for i=1:centernum
    cos2Theta{i} = zeros(10,10);
    sin2Theta{i} = zeros(10,10);
    cospart = OF_center(i,1:dim/2);
    cospart = reshape(cospart,10,10);
    sinpart = OF_center(i,dim/2+1:end);
    sinpart = reshape(sinpart,10,10);
    ori = atan2(sinpart,cospart);
    cos2Theta{i} = cos(ori);
    sin2Theta{i} = sin(ori);
end

% [centernum,dim] = size(RF_center);
% 
% frecenter_cell = cell(centernum,1);
% for i=1:centernum
%     fpatch = RF_center(i,:);
%     frecenter_cell{i} = reshape(fpatch,10,10);
% end

imglist = dir([pathname '*.bmp']);
blksize = 16;
sigma = 2;
oimg = cell(length(imglist),1);

load('data/OFClassNo_128/Results/OF_STFT_Enh_5.mat','oimg')
% matlabpool open
parpool(24)
tic
for i=1:length(imglist)
    i
    img0 = imread([pathname imglist(i).name]);
    mask = imread([maskpath 'roi' imglist(i).name]);
    blkmask = mask(8:16:end,8:16:end);

    fname = ['data/OFClassNo_128/Enhancement_gabor_OF_RF_2/' imglist(i).name ];
%     [oimg{i},fimg{i}] = GetOFRFViaDP_Pool(OF_net,cos2Theta,sin2Theta,RF_net,frecenter_cell,img0,mask,fname,oimg{i});


%     [oimg{i},fimg{i}] = GetOFRFViaDeepLearning(img0,mask,OF_net,RF_net,cos2Theta_c,sin2Theta_c,fcenter,opts)
% 
% tic
    [oimg{i}] = GetOrientationFieldViaDP_Pool(net,cos2Theta,sin2Theta,img0,mask);
%     toc
% %     
%     Show(2,img0,'Orientation Field');
%      
%     DIR = -oimg{i}*180/pi;
%     DIR(blkmask==0) = 91;
%     Obj = DrawDir(2,DIR,blksize,'r');
% %     saveas(gcf,['OF_STFT_Enh/' imglist(i).name(1:end-3) 'tif'],'tif');
%     keyboard
end
toc
%  matlabpool close
% save('data/OFClassNo_128/Results/OF_FF_STFT_Enh_weight.mat','oimg','fimg');