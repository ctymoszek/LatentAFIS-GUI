function minutiae_cylinder = extract_minutiae_cylinder(img,minutiae)
[h,w] = size(img);
num_ori = 12;
sigma = 5^2;
minutiae_cylinder = zeros(h,w,12);
[X,Y] = meshgrid(1:w,1:h);
cylinder_ori = (0:num_ori-1)*pi*2/num_ori;
for i=1:size(minutiae,1)
    x = minutiae(i,1);
    y = minutiae(i,2); 
    weight = exp(-((X - x) .* (X - x) + (Y - y).* (Y - y)) / sigma);
    ori = minutiae(i,3);
    if ori<0
        ori = ori+pi*2;
    end
    for j=1:num_ori
        ori_diff = abs(ori - cylinder_ori(j));
        ori_diff = min(ori_diff,2*pi-ori_diff); 
        minutiae_cylinder(:,:,j) = minutiae_cylinder(:,:,j)+weight.* exp(-ori_diff/pi*12);
    end
    
end

minutiae_cylinder = minutiae_cylinder*255;
minutiae_cylinder(minutiae_cylinder>255) = 255;

% keyboard