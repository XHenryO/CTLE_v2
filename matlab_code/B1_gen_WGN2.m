close all;
clear all; 
% this program is used to generate .wfm 
%% 产生训练信道的发射数据
%% Global variable
QAMorder        = 4;
BitsPerSymbol   = log2(QAMorder);
upsampleno      = 4;
carrierno       = 512;
datano          = 100;
AWGSampleRate   = 100e3;
OSCSampleRate   = 100e3;
CP              = 0/32*carrierno;%1/32
trainingno      = 20;
zeropad1        = 2;
zeropad2        = 0;
carrierno1      = carrierno/2-zeropad1-zeropad2;
DFTS_flag       = 0;

%%
% rng(20)

name= './data/WGN';
% name11= 'Z:/JY/WGN';

data_len=carrierno*datano*upsampleno;  %% 这里待定

zhengchu=128;   %% 根据AWG来定
% length_zero=2^(floor(log2(length(data1)))+1)-length(data1);
data_len=(floor(data_len/zhengchu)+1)*zhengchu;

data1=rand(data_len,1);
data1=(data1-0.5)*2;  %%
rms_val = sqrt(mean(data1.^2));
data1 = data1 / rms_val;
% data1(1)=1;
% data1(2)=-1;
% data1(data_len/4:data_len/4*2)=data1(data_len/4:data_len/4*2)*0.8;
% data1(data_len/4*2:data_len/4*3)=data1(data_len/4*2:data_len/4*3)*0.5;
% data1(data_len/4*3:data_len/4*4)=data1(data_len/4*3:data_len/4*4)*0.3;

zhengchu=96;   % 96 for M8190 14bit; 128 for M8195; 64 for 12bit M8190  投稿1用的64
data_len=length(data1);
data_len=(floor(data_len/zhengchu)+1)*zhengchu;
len_zero=data_len-length(data1);
data1=[data1;zeros(len_zero,1)];

% data1=data1./max(abs(data1));
% figure();
% plot(data1);
% xlabel('Samples', 'FontName', 'Times New Roman', 'FontSize', 14, 'FontWeight', 'bold');  % 设置x轴标题为粗体
% ylabel('Amplitude', 'FontName', 'Times New Roman', 'FontSize', 14, 'FontWeight', 'bold');  % 设置y轴标题为粗体
% set(gca, 'FontName', 'Times New Roman', 'FontSize', 14, 'LineWidth', 1);
% set(gcf,'unit','centimeters','position',[10 5 13 8.67]);      %% 双栏36 12; 单栏15 10   3:2
% print('./draw_oe_bushiyan/b_sample', '-dpng', '-r300');  % '-r600' 指定图像的分辨率为600 DPI
% axis off;


% xfre=[(0:1:length(data1)-1)-length(data1)/2]*AWGSampleRate/1e9/length(data1);
% figure();plot(xfre,20*log10(abs(fftshift(fft(data1(:))))),'b.-');title('Noise');

eval(['save ' name '.txt data1 -ASCII -double']);
% save(['./data/'])
eval(['disp(''File Name is  ' name '.txt '')']);

% eval(['save ' name11 '.txt data1 -ASCII -double']);
% eval(['disp(''File Name is  ' name11 '.txt '')']);

% % Store as wfm
% % place = '../Store/20210805/AWG/';%for Mac
% place = '..\Store\20220113\AWG\';%for Windows
% place= '';
% makewfm(data1,[place,name,'.wfm'])   % Make .wfm

% factor = 10;           % 衰减系数
% SNR = 25;              % SNR
% vpp=1;
% dataRx=z_ch_simu(data1,vpp,SNR,factor);

bandwidth_scale = 100;     % synthetic frequency-axis scale
attenuation_factor = 30;   % larger means less high-frequency attenuation
                 % used only when enable_awgn is true
enable_nonlinear = false;
enable_awgn = false;
enable_extra_isi = false;
dataRx = VLC_channel_waveform(data1, bandwidth_scale, attenuation_factor);

% figure();plot(xfre,20*log10(abs(fftshift(fft(dataRx)))),'b.-');title('simu Rx Signal');
data_store_for_Test=resample(dataRx,OSCSampleRate,AWGSampleRate);
save D://data/SimulationDateAfterChannel_B1.txt data_store_for_Test -ASCII;
