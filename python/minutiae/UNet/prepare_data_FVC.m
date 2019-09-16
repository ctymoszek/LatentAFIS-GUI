pathname{1} = '/media/kaicao/Data/MinutiaeExtraction/image/FVC2002/DB1_A/img/';
minu_path{1} = '/media/kaicao/Data/MinutiaeExtraction/minutiae/FVC2002/DB1_A/';
prefix{1} = 'FVC2002_DB1A_';
% data_path = '/media/kaicao/Data/AutomatedLatentRecognition/minutiae_cylinder_uint8_FVC_mat/';
data_path = '/media/kaicao/data2/AutomatedLatentRecognition/Data/minutiae_FVC/';
mkdir(data_path)

pathname{2} = '/media/kaicao/Data/MinutiaeExtraction/image/FVC2002/DB3_A/img/';
minu_path{2} = '/media/kaicao/Data/MinutiaeExtraction/minutiae/FVC2002/DB3_A/';
prefix{2} = 'FVC2002_DB3A_';

pathname{3} = '/media/kaicao/Data/MinutiaeExtraction/image/FVC2004/DB1_A/img/';
minu_path{3} = '/media/kaicao/Data/MinutiaeExtraction/minutiae/FVC2004/DB1_A/';
prefix{3} = 'FVC2004_DB1A_';


pathname{4} = '/media/kaicao/Data/MinutiaeExtraction/image/FVC2004/DB3_A/img/';
minu_path{4} = '/media/kaicao/Data/MinutiaeExtraction/minutiae/FVC2004/DB3_A/';
prefix{4} = 'FVC2004_DB3A_';

show_minutiae = 0;
for n =1:length(pathname)
    
    img_files = dir([pathname{n} '*.tif']);
    minu_files = dir([minu_path{n} '*iso*']);
    
    R = 15;
    for i=1:length(img_files)
        outname = [data_path prefix{n} img_files(i).name(1:end-3) 'mat'];
%         if exist(outname,'file')==2
%             continue;
%         end
        img = imread([pathname{n} img_files(i).name]);
        T = LoadMinutiaeFromISO([minu_path{n} minu_files(i).name]);
        minutiae = [T.MntXY T.MntAngle];

        %
        if show_minutiae
            close all
            figure(1)
            imshow(img);
            hold on
            plot(minutiae(:,1),minutiae(:,2),'ro','MarkerSize',10)
            for j=1:size(minutiae,1)
                x = minutiae(j,1);
                y = minutiae(j,2);
                ori = minutiae(j,3);
                x1 = x+R*cos(-ori);
                y1 = y+R*sin(-ori);
                plot([x,x1],[y,y1],'r-')
            end
        end
        f = fspecial('gaussian',20,5);
        nimg = imfilter(img,f,'replicate');
        nimg = double(nimg);
        col_mean  = mean(nimg);
        ind = find(col_mean<254);
        if isempty(ind)
            keyboard
        end
        minx = min(ind);
        maxx= max(ind);
        
        row_mean  = mean(nimg,2);
        ind = find(row_mean<254);
        if isempty(ind)
            keyboard
        end
        miny = min(ind);
        maxy = max(ind);
        
        if maxy - miny<128 | maxx-minx <128
            continue;
        end
        
        img = img(miny:maxy,minx:maxx);
        minutiae(:,1) = minutiae(:,1) - minx;
        
        minutiae(:,2) = minutiae(:,2) - miny;
        
        if show_minutiae
            figure(2)
            imshow(img);
            hold on
            plot(minutiae(:,1),minutiae(:,2),'ro','MarkerSize',10)
            for j=1:size(minutiae,1)
                x = minutiae(j,1);
                y = minutiae(j,2);
                ori = minutiae(j,3);
                x1 = x+R*cos(-ori);
                y1 = y+R*sin(-ori);
                plot([x,x1],[y,y1],'r-')
            end
        end
        
    
         
        save([data_path prefix{n} img_files(i).name(1:end-3) 'mat'],'img','minutiae')
        continue;
        
        minutiae_cylinder = extract_minutiae_cylinder(img,minutiae);
       
        
        ROI = zeros(size(img))+255;
        
        
        save([data_path prefix{n} img_files(i).name(1:end-3) 'mat'],'img','ROI','minutiae_cylinder')
        %     keyboard
    end
end