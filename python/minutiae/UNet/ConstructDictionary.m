function [D,Feature,D_Ori,Feature_Ori] = ConstructDictionary(varargin)


oriNum = 120;

len = 16;
x = -len+1:len;
y = -len+1:len;

[X,Y] = meshgrid(x,y);
N = 20000;
D = zeros(length(x)^2,N);
% OriField = zeros(1,N);
Feature = zeros(4,N);
% 1st row: orienation
% 2nd row: ridge spacing
% 3rd row: valley spacing
% 4th row: offset
n = 0;
for spacing = 6:14
    
    for valley_spacing =3:1:spacing/2+0.5
        ridge_spacing = spacing - valley_spacing;
        
        if ridge_spacing <3
            continue;
        end
        
        for k = 1:oriNum
            
            theta = (k-1)*pi/oriNum;
            X_r = X*cos(theta) - Y*sin(theta);
            %     Y_1 = X*sin(theta) + Y*cos(tehta);
            %             spacing = ridge_spacing + valley_spacing;
            
%             figure(1),imshow(cos(2*pi*(X_r/spacing)),[])
            
            for offset = 0:1:spacing-1
                
                X_r_offset = X_r + offset + ridge_spacing/2;
                
                
                X_r_offset = mod(X_r_offset,spacing);
                
                Y1 = zeros(size(X_r_offset));
                Y2 = zeros(size(X_r_offset));
                
                
                ind = X_r_offset<ridge_spacing;
                Y1(ind==1) = X_r_offset(ind==1);
                Y2(ind==0) = X_r_offset(ind==0)-ridge_spacing;
                element = -sin(2*pi*(Y1/ridge_spacing/2))+sin(2*pi*(Y2/valley_spacing/2));
                n = n + 1;
                
                Feature(1,n) = k;
                Feature(2,n) = ridge_spacing;
                Feature(3,n) = valley_spacing;
                Feature(4,n) = offset;
                Feature(5,n) = element(len+1,len+1);
                
              
                if offset == 0  % ridge
%                       Feature(5,n) 
                      Feature(6,n) = -1;
                     
                elseif offset == spacing/2  % valley
%                      Feature(5,n) 
                    Feature(6,n) = 1;
                    
                else
                     Feature(6,n) = 0;
                end
                
                Feature(7,n) = theta;
                Feature(8,n) = spacing;
%                 figure(2),imshow(cos(2*pi*(X_r_offset/spacing)),[])
%                 figure(4),imshow(element,[])

                
                element = element(:);
                
                
                
                element = element - mean(element);
                
                element = element/(norm(element)+0.0001);
                D(:,n) = element(:);
                OriFied(n) = theta;
                
                
%                 figure(3),imshow(-sqrt(sin(2*pi*(Y1/ridge_spacing/2)))+sqrt(sin(2*pi*(Y2/valley_spacing/2))),[])
             end
            
        end
    end
end

if n<N
    D = D(:,1:n);
    Feature = Feature(:,1:n);
end

D_Ori = cell(oriNum,1);
Feature_Ori = cell(oriNum,1);
for i = 1:oriNum
    ind = find(Feature(1,:)==i);
    D_Ori{i} = D(:,ind);
    Feature_Ori{i} = Feature(:,ind);
end