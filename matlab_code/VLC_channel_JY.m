function data_after=VLC_channel_JY(data1,carrierno,upsampleno,CP,Fs,factor,SNR,process_nonlinear)

n=1:(carrierno+CP)*upsampleno/2; 
fs=2*Fs/((carrierno+CP)*upsampleno);
fs=fs*n;
ch1=exp(-fs/factor);
ch2=fliplr(ch1);
if mod((carrierno+CP)*upsampleno,2)==1
    ch=[ch1 1 ch2].';
else
    ch=[ch1 ch2].';
end
% chall=repmat(ch,1,datano);
index_f_use=1:length(ch);
index_f_use_right=linspace(1,length(ch),length(data1));
chall=spline(index_f_use,ch,index_f_use_right);
chall=reshape(chall,[],1);

% process_nonlinear = 1;
if process_nonlinear == 1
    nonlinear_flag = 1; % 非线性
    ISI_flag = 0; % ISI设置
else
    nonlinear_flag = 0; % 非线性
    ISI_flag = 0; % ISI设置
end

% 非线性
b1 = 1.259;
b2 = 0.01373;
b3 = -0.4576;
divide = 0.8165;%0.8165
if nonlinear_flag == 1
    data_tx_temp = abs(data1/max(abs(data1)));
    data_rx_temp = (b1*data_tx_temp+b2*data_tx_temp.^2+b3*data_tx_temp.^3)/divide;%0.01373
    % data_rx_temp = (b1*data_tx_temp+b2*data_tx_temp.^2)/divide;
    x = 0:0.01:1;
    y = (b1*x+b2*x.^2+b3.*x.^3)/divide;
    % y = (b1*x+b2*x.^2);
    figure,plot(x,y,'b','Linewidth',3),title('AM Response')
    hold on;
    plot(x,x,'r','Linewidth',3)
    axis([0 1 0 1])
    dataTx = sign(data1/max(abs(data1))).*data_rx_temp;
    dataTx = dataTx/2;
else
    dataTx = data1;
end
% figure,plot(dataTx,'b.-')

atte_flag = 1;
% dataTx=reshape(dataTx,(carrierno+CP)*upsampleno,datano);
data_xindao_fft = (fft(dataTx));
%data_xindao_fft(1,:) = 0;% 假设去掉0频的分量
if atte_flag == 1
    data_xindao_after=data_xindao_fft.*chall;
else
    data_xindao_after=data_xindao_fft;
end
data_xindao_ifft=(ifft(data_xindao_after));
% mean_tx_power = sum(abs(data_xindao_ifft).^2)/(carrierno+CP)*upsampleno;
% figure,plot(mean_tx_power);

% ISI设置
taps = 5;%指示当前的taps数
w = [0.22,-0.6,1,-0.6,+0.22];
% taps = 53;
% w = importdata('W.txt');
if ISI_flag == 1
    data_xindao_ifft_new = data_xindao_ifft;
%     for i = 1:datano
%         for j = floor(taps/2)+1:(carrierno+CP)*upsampleno-floor(taps/2)
%             data_xindao_ifft_new(j,i)=w*data_xindao_ifft_new(j-floor(taps/2):j+floor(taps/2),i);
%         end
%     end
    data_xindao_ifft_new = conv(data_xindao_ifft_new,w,'same');
    data_xindao_ifft = data_xindao_ifft_new;   
end

awgn_flag = 0;
rng(110)
if awgn_flag == 1
    dataRx=awgn(data_xindao_ifft,SNR,'measured');
else
    dataRx = data_xindao_ifft;
end
data_after=real(reshape(dataRx,[],1)); 
end