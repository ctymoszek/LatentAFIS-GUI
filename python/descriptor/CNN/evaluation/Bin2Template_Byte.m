function template = Bin2Template_Byte(fname,isLatent)
fname
fid = fopen(fname,'rb');
% image size

if fid<=0
    error(['template file' fname 'could not be open']);
end
tmp = fread(fid,2,'uint16');
h = tmp(1);
w = tmp(2);

if h==0 || w == 0
    template = [];
    fclose(fid);
    return;
end

tmp = fread(fid,2,'uint16');
blkH = tmp(1);
blkW = tmp(2);



% template 1
% template.minutiae = minutiae;
% template.Des = Des;
% template.oimg = oimg;
% template.mask = mask;

% minutiae points
num_template= fread(fid,1,'uint8');
minu_template = cell(num_template,1);
for n = 1:num_template
    minu_num = fread(fid,1,'uint16');
    minutiae = zeros(minu_num,3);
    if minu_num >0
         x = fread(fid,minu_num,'uint16'); % x
         minutiae(:,1) = x;
         y = fread(fid,minu_num,'uint16'); % x
         minutiae(:,2) = y;
         ori = fread(fid,minu_num,'single'); % x
         minutiae(:,3) = ori;
         des_num = fread(fid,1,'uint16');
         des_len = fread(fid,1,'uint16');
         Des = cell(des_num,1);
         for i=1:des_num
             tmp = fread(fid,des_len*minu_num,'uint16'); % descriptor
             tmp = reshape(tmp,des_len,minu_num);
             for j=1:minu_num
                 tmp(:,j) = tmp(:,j)/(norm(tmp(:,j))+0.0001);
             end
             Des{i} = tmp;
         end
     else
        Des = [];
     end
    oimg = fread(fid,blkH*blkW,'single');
    oimg = reshape(oimg,blkH,blkW);

    run_mask_num = fread(fid,1,'uint16');
    run_mask = fread(fid,run_mask_num,'uint32');
    mask = RunLengthEncoding(run_mask,h,w);
    % figure(1);imshow(template_1.mask,[])
    %figure(2);imshow(mask,[])
    minu_template{n}.minutiae = minutiae;
    minu_template{n}.Des= Des;
    minu_template{n}.oimg = oimg;
    minu_template{n}.mask= mask;

end


% texture template
% minutiae points
minu_num = fread(fid,1,'uint16');
minutiae = zeros(minu_num,4);
if minu_num >0
     x = fread(fid,minu_num,'uint16'); % x
     minutiae(:,1) = x;
     y = fread(fid,minu_num,'uint16'); % x
     minutiae(:,2) = y;
     ori = fread(fid,minu_num,'single'); % x
     minutiae(:,3) = ori;
     if isLatent == 1
         D = fread(fid,minu_num,'single'); % x
         minutiae(:,4) = D;
     end
     des_num = fread(fid,1,'uint16');
     des_len = fread(fid,1,'uint16');
     Des = cell(des_num,1);
     for i=1:des_num
         tmp = fread(fid,des_len*minu_num,'uint16'); % descriptor
         tmp = reshape(tmp,des_len,minu_num);
         for j=1:minu_num
             tmp(:,j) = tmp(:,j)/(norm(tmp(:,j))+0.0001);
         end
         Des{i} = tmp;
     end
     
end
texture_template.minutiae = minutiae;
texture_template.Des= Des;
texture_template.mask = minu_template{1}.mask;

template.minu_template = minu_template;
template.texture_template = texture_template;

fclose(fid);




