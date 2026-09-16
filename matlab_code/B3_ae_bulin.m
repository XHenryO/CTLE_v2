%% AE补零
clear;
close all;

%% 参数

carrierno       = 512;
datano          = 100;

upsampleno=3;
AWGSampleRate=6e9; 
BaudRate=AWGSampleRate/upsampleno;
% OSCSampleRate=AWGSampleRate;
OSCSampleRate=20e9;

CP              = 0/32*carrierno;%1/32
trainingno      = 20;
zeropad1        = 2;
zeropad2        = 0;
carrierno1      = carrierno/2-zeropad1-zeropad2;
DFTS_flag       = 0;

%% 读取文件
data_tx=importdata('E:/experiment/ae_compare/ae/txdata/Txdata_Tx.txt');

%% 重采样   %% 貌似不需要

zhengchu=96;   % 96 for M8190 14bit; 128 for M8195; 64 for 12bit M8190  投稿1用的64  补实验 96
data_len=length(data_tx);
data_len=(floor(data_len/zhengchu)+1)*zhengchu;
len_zero=data_len-length(data_tx);
data_tx_fianl=[data_tx;zeros(len_zero,1)];
figure();
plot(fftshift(10*log10(abs(fft(data_tx_fianl)))));
disp('len zero');
disp(len_zero);
title("tx final fft");

% figure();
% plot(data_tx_fianl(1:300),'LineWidth',2,'Color','#92bcdb');
% % title('发送波形');
% axis off;
%% 保存波形
save('E:/experiment/ae_compare/ae/txdata/Txdata_Tx_final.txt','data_tx_fianl','-ascii');

%% 经过信道
% Fs = 100;              % 带宽
% factor = 30;           % 衰减系数
% SNR = 59;              % SNR
% process_nonlinear = 0; % 非线性处理标志位
% 
% figure();
% plot(fftshift(10*log10(abs(fft(data_tx_fianl)))));
% title("tx fft");
% dataRx=VLC_channel_JY(data_tx_fianl,carrierno,upsampleno,CP,Fs,factor,SNR,process_nonlinear);
% data_store_for_Test=dataRx;

factor = 30;           % 衰减系数
SNR = 25;              % SNR
vpp=1;
dataRx=z_ch_simu(data_tx_fianl,vpp,SNR,factor);
data_store_for_Test=dataRx;

% data_store_for_Test=resample(dataRx,OSCSampleRate,AWGSampleRate);
save SimulationDateAfterChannel_B3.txt data_store_for_Test -ASCII;

figure();
plot(fftshift(10*log10(abs(fft(dataRx)))));
disp('len zero');
disp(len_zero);
title("simu Rx fft");

figure();
%% data_tx_fianl    dataRx
plot(dataRx(1:200),'LineWidth',5,'Color','#92bcdb');
% title('发送或者接收波形');
axis off;

figure();
plot(data_tx,'.');
data_tx=data_tx/max(data_tx);
a_x=linspace(0,65472,65472)';