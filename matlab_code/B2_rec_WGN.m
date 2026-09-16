close all;
clear;
% clc
% this program is used to evaluate the performance of OFDM and DFTS-OFDM
%% 接收训练信道的数据
addpath('./data/')
addpath('./function/')
%% Global variable
QAMorder        = 4;
BitsPerSymbol   = log2(QAMorder);
% upsampleno      = 2;
carrierno       = 512;
datano          = 100;
AWGSampleRate   = 100e9;
OSCSampleRate   = 100e9;
CP              = 0/32*carrierno;%1/32
trainingno      = 20;
zeropad1        = 8;
zeropad2        = 0;
carrierno1      = carrierno/2-zeropad1-zeropad2;
DFTS_flag       = 0;

CP = 0;
%%
origidate=load(['D:/git_CTLE/CTLE_v2/data/WGN.txt']);
dfts_data=origidate;
%% 示波器采集数据
ONLINE_FLAG = 0; % 1：online  0：offline
Simulation_FLAG = 1; % 1: Simulation
index11 = 97; % 0: Last Data 
GPIB_flag = 0;
h=1;
channel_num = 1;
Channel_no1 = 1;
Channel_no2 = 2;
Channel_no3 = 3;
Channel_no4 = 3;
switch Simulation_FLAG
    case 1
        datatx=load('D:/git_CTLE/CTLE_v2/data/WGN.txt');
        dataRx=load('D:/git_CTLE/CTLE_v2/data/SimulationDateAfterChannel_B1.txt');
        Origin_filename='DMT_4QAM_zero_8_zero_0.txt';
        dataRx=resample(dataRx,AWGSampleRate,OSCSampleRate);
    otherwise
        filename = ['./Rx/oscWGN_'];
        [ data1,data2,data3,data4 ] = OSC_receive( ONLINE_FLAG,filename,index11,OSCSampleRate,GPIB_flag,channel_num,Channel_no1,Channel_no2,Channel_no3,Channel_no4);

        %
        %同步取出信号

        Origin_filename =  ['WGN.txt'];
%         Origin_filename=['E:/experiment/ae_compare/ae/txdata/Txdata_Tx_final.txt'];

        [dataRx,datatx] = Synchronization(Origin_filename,AWGSampleRate,OSCSampleRate,channel_num,data1,data2,data3,data4,h);
end
data_t=dfts_data;
% data1=dataRx(end-(CP+carrierno)*datano*upsampleno+1:end);
data1=dataRx(end-length(dfts_data)+1:end);
%% Nonlinearity
data_real=data_t/sqrt(mean(abs(data_t).^2));
data_afterchannel=-data1/sqrt(mean(abs(data1).^2));  %% 功率归一化
data_len=length(data_real);
data_real=data_real(data_len/4*0+1:data_len/4*1);
data_afterchannel=data_afterchannel(data_len/4*0+1:data_len/4*1);
data_real=data_real/sqrt(mean(abs(data_real).^2));
data_afterchannel=data_afterchannel/sqrt(mean(abs(data_afterchannel).^2))*2; %% 功率归一化
figure;plot(data_real,-data_afterchannel,'.');hold on;plot(data_real,data_real);title('Nonlinearity');

% plotAMAM_JY(data_real,data_afterchannel)
% in = reshape(data_real,1,[]);
% out = reshape(data_afterchannel,1,[]);
% plot_hist3nan(1j*in+out);hold on;axis([-1,1,-1,1]);alpha(0.5);
%%
% load Rx_Data.txt;
% data=Rx_Data(15873:end);
data_rx=data1(1:end);
xfre=[(0:1:length(data_rx)-1)-length(data_rx)/2]*AWGSampleRate/1e9/length(data_rx);
figure();plot(xfre,20*log10(abs(fftshift(fft(data_rx)))),'b.-');title('Rx Signal');
%% filter
% index_13G = find(xfre<=-13);
% index_13G = index_13G(end);
% data_rx_fft=fftshift(fft(data_rx));
% data_rx_fft_filter=data_rx_fft;
% data_rx_fft_filter(1:index_13G)=0;
% data_rx_fft_filter(end-index_13G+1:end)=0;
% data_rx=real(ifft(fftshift(data_rx_fft_filter)));
% % data_rx=data_rx-mean(data_rx);
% figure();plot(xfre,20*log10(abs(fftshift(fft(data_rx)))),'b.-');title('Rx Signal after Filter');
% 
% data_tx_fft=fftshift(fft(data_t));
% data_tx_fft_filter=data_tx_fft;
% data_tx_fft_filter(1:index_13G)=0;
% data_tx_fft_filter(end-index_13G+1:end)=0;
% Txdata=real(ifft(fftshift(data_tx_fft_filter)));
%% 
Rxdata=data_rx;
Txdata=real(data_t);

% Txdata=importdata('E:/experiment/ae_compare/ae/txdata/Txdata_Tx_final.txt');

disp('NN')
save D:/git_CTLE/CTLE_v2/data/Rxdata_NN_baseline_wgn.txt Rxdata -ASCII;
save D:/git_CTLE/CTLE_v2/data/Txdata_NN_baseline_wgn.txt Txdata -ASCII;
