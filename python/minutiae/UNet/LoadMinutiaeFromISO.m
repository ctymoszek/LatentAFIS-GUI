% 从ＩＳＯ模板中读取数据，然后初始化一个FPTemplate模板
% 
% 函数声明：
% T = LoadFPTemplateFromISO(file)
%
% 描述：
% 该函数实现ISO模板文件的读取功能，关于ISO模板的详细定义
% 请看Doc文件夹中的ISOTemplate.pdf 及
% SC37-19794-2FDIS.pdf文件。ISO模板查看器请看ISOTemplateView.exe文件
% 
% 调用：
% T = LoadFPTemplateFromISO(file)
% 
% 输入：
% file - iso模板文件名
%
% 输出：
% 读取的细节点数据，细节点质量，等等都保存到对应的属性中
% 具体读取的特征还要看ISO中保存有什么特征
% 
% 例子：
% T = LoadFPTemplateFromISO('testimages/fing.ist');
% T.Img = imread('testimages/fing.bmp');
% ShowMnt(T);
%
% SEE ALSO SaveFPTemplateToISO, ReadISOUserDefinedExtendData 
%
function T = LoadMinutiaeFromISO(file)

% 打开文件
fid = fopen(file, 'r');
if fid == -1
    error(['无法打开ISO模板文件: ' file]);
end

%读取数据
%%
% 读取文件头
[ID, Version, RecLen, CEC, CDT, ImSize, HRes, VRes, NumView] = readisofileheader(fid);
if(strcmp(ID(1:3), 'FMR')~=1)
    error(['文件' file '是一个非ISO细节点模板文件']);
end

% 读取每一个finger View的数据
% 将文件指针放置到指向第一个view
fseek(fid, 1, 'cof');
% T = FPTemplate(1,NumView);% 预先初始化一个指纹模板数组，数组大小就是NumView
for k=1:NumView
    T(k) = readviewdata(fid);
    T(k).ImageSize = ImSize;
    T(k).HRes = HRes;
    T(k).VRes = VRes;
end

% 关闭文件
fclose(fid);

if nargin == 0
    % 只显示第一个view
%     T(1).Img = im;
%     ShowMnt(T(1),'shownumber', 'on');
%     hold on;ShowSP(T(1));hold off;
end

function [ID, Version, RecLen, CEC, CDT, ImSize, HRes, VRes, NumView] = readisofileheader(fid)
%
% 读取文件头
% 
% 模板文件标识符
fseek(fid, 0, 'bof');
ID = fread(fid, 4, '*char')';

% 读取版本号
Version = fread(fid, 4, '*char')';

% 读取模板总长度
RecLen = fread(fid, 1, 'uint32', 0, 'b');

% 读取采集设备认证码和采集设备ID号
code = fread(fid, 1, 'uint16', 0, 'b');
c = dec2bin(code, 16);
CEC = bin2dec(c(1:4));  % 前四个比特表示Capture Equipment Certification
CDT = bin2dec(c(5:16)); % 后12比特表示采集仪生产商标识符

% 读取图像尺寸
ImSize = fread(fid, 2, 'uint16', 0, 'b');
ImSize = ImSize(end:-1:1)';

% 读取图像分辨率
HRes = fread(fid, 1, 'uint16', 0, 'b');
VRes = fread(fid, 1, 'uint16', 0, 'b');

% 读取Number of Views
NumView = fread(fid, 1, 'uint8', 0, 'b');


function [T, ViewNumber] = readviewdata(fid)
%
% 读取文件中的每一个view的数据，并存入T模板
% 文件标识符fid必须已经放置在view的数据开头
% 

% 读取Position数据
T.Position = fread(fid, 1, 'uint8',0,'b');

% 读取ViewNumber和Impression Type
c = fread(fid, 1, 'uint8', 0, 'b');
c = dec2bin(c, 8);
ViewNumber = bin2dec(c(1:4));
T.ImpressionType = bin2dec(c(5:8));

% 读取细节点质量
T.Quality = fread(fid, 1, 'uint8', 0, 'b');

% 细节点数量
NumMnt = fread(fid, 1, 'uint8', 0, 'b');

% 逐个读取细节点数据
MntXY = zeros(NumMnt,2);
MntAngle = zeros(NumMnt,1);
MntType = zeros(NumMnt,1);
MntQuality = zeros(NumMnt,1);
for k=1:NumMnt
    c = fread(fid, 1, 'uint16', 0, 'b');c = dec2bin(c,16);
    MntType(k) = bin2dec(c(1:2));
    MntXY(k,1) = bin2dec(c(3:16));
    c = fread(fid, 1, 'uint16', 0, 'b');c = dec2bin(c,16);
    MntXY(k,2) = bin2dec(c(3:16));
    
    c = fread(fid, 2, 'uint8', 0, 'b');
    MntAngle(k) = c(1);
    MntQuality(k) = c(2);    
end
T.MntXY = MntXY;
T.MntAngle = mod(MntAngle * 2*pi/255, 2*pi);  % 将角度转换为Matlab的常用格式
T.MntType = MntType;
T.MntQuality = MntQuality;


%% 以下读取扩展数据


% 扩展数据区长度
ExtDataLen = fread(fid, 1, 'uint16', 0, 'b');
if ExtDataLen == 0,return;end % 为0说明没有扩展数据，直接退出

% 扩展数据的类型由类型码来标识，类型标识码占两个字符，各种数据的标识码如下表所示：
%---------------------------------------------------------------%
% First byte    Second byte     Identification
% 0x00          0x00            reserved
% 0x00          0x01            ridge count data (7.5.2)
% 0x00          0x02            core and delta data (7.5.3)
% 0x00          0x03            zonal quality data (7.5.4)
% 0x00          0x04-0xFF       reserved
% 0x01-0xFF     0x00            reserved
% 0x01-0xFF     0x01-0xFF       vendor-defined extended data
%----------------------------------------------------------------%
% 要在ISO模板中支持自己定义的扩展数据，则需要自己定义一些类型标识码
% 不同的扩展数据，其存储格式也不一样，所以需要针对不同的扩展数据写不同的读取函数
% 以下程序循环读取扩展数据，直到所有数据都读取进来，遇到不支持的扩展数据则忽略，并给出警告！！

% 记录下此时文件指针的位置
PositionExtData = ftell(fid);
while 1 %由于不知道有几个扩展数据，必须在循环中检查总的扩展数据长度，然后用break跳出
    % 如果已经读取的扩展数据长度等于扩展数据总长度，则退出循环
    PositionNow = ftell(fid);
    if PositionNow-PositionExtData==ExtDataLen,break;end
    
     % 首先读取扩展数据标识码
    ExtDataID = fread(fid, 1, 'uint16', 0, 'b');
    switch ExtDataID
        case 0      %保留的数据
            error('扩展数据类型标识0为保留数字，请检查文件是否损坏！');
        case 1      % Ridge count 数据
            [MntRidgeCount, Method] = readisoridgecount(fid);
            T.MntRidgeCount = MntRidgeCount;
            T.MntRidgeCountMethod = Method;
        case 2      % 奇异点数据
            % 扩展区的长度
            AreaDataLen = fread(fid, 1, 'uint16', 0, 'b');
            [core, delta, coreinfo, deltainfo] = readisosingular(fid);
            T.CoreXY = core(:,1:2);
            if coreinfo==1,T.CoreAngle = core(:,3);end
            T.DeltaXY = delta(:,1:2);
            if deltainfo==1,T.DeltaAngle = delta(:,3:end);end
        case 3      % zonal质量数据
            ;
        otherwise   % 否则调用自己定义的函数来读取
            T = ReadISOUserDefinedExtendData(fid, ExtDataID, T);
    end
end

function [MntRidgeCount, Method] = readisoridgecount(fid)
%
%
% 数据区长度
AreaLen = fread(fid, 1, 'uint16', 0, 'b');

% 计算总共的脊线计数特征数量
NumPair = (AreaLen-1)/3;
MntRidgeCount = zeros(NumPair, 3);

% 读取细节点脊线计数特征
Method = fread(fid, 1, 'uint8', 0, 'b');

% 逐对读取
for k=1:NumPair
    MntRidgeCount(k,1:3) = fread(fid, 3, 'uint8', 0, 'b');
end


function [core, delta, coreinfo, deltainfo] = readisosingular(fid)
% 读取core点
c = fread(fid, 1, 'uint8',0,'b');c = dec2bin(c, 8);
NumCore = bin2dec(c(3:8));
core = zeros(NumCore, 3);
for k = 1:NumCore
    c = fread(fid, 1, 'uint16', 0, 'b');c = dec2bin(c,16);
    coreinfo = bin2dec(c(1:2));
    core(k,1) = bin2dec(c(3:16));
    c = fread(fid, 1, 'uint16', 0, 'b');c = dec2bin(c,16);
    core(k,2) = bin2dec(c(3:16));
    if coreinfo==1 % 包含角度信息
        angle = fread(fid,1,'uint8', 0, 'b');
        core(k,3) = mod(-angle*2*pi/255, 2*pi);
    end
end

% 读取Delta点
c = fread(fid, 1, 'uint8',0,'b');c = dec2bin(c, 8);
NumDelta = bin2dec(c(3:8));
delta = zeros(NumDelta, 5);
for k = 1:NumDelta
    c = fread(fid, 1, 'uint16', 0, 'b');c = dec2bin(c,16);
    deltainfo = bin2dec(c(1:2));
    delta(k,1) = bin2dec(c(3:16));
    c = fread(fid, 1, 'uint16', 0, 'b');c = dec2bin(c,16);
    delta(k,2) = bin2dec(c(3:16));
    if deltainfo==1 % 包含角度信息
        angle = fread(fid,3, 'uint8', 0, 'b');
        delta(k,3:5) = mod(-angle*2*pi/255, 2*pi);
    end
end