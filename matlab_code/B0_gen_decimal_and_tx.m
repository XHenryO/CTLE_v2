% CAP 生成代码
% 产生训练autopre的数据
clear ;
close all;
%%参数
QAMorder=4;

upsampleno=4;
AWGSampleRate=200e9; 
BaudRate=AWGSampleRate/upsampleno;
% OSCSampleRate=AWGSampleRate;
OSCSampleRate=200e9;

filtersequence=16;
zhengchu=64;   % 96 for M8190 14bit;  128 for M8195;64 for 12bit M8190
numofsymbols    =(floor(2^16/lcm(zhengchu*BaudRate/gcd(BaudRate,AWGSampleRate),AWGSampleRate/gcd(BaudRate,AWGSampleRate)))*lcm(zhengchu*BaudRate/gcd(BaudRate,AWGSampleRate),AWGSampleRate/gcd(BaudRate,AWGSampleRate)))/(AWGSampleRate/1e9)*(BaudRate/1e9);
rolloff=0.5; % line13 in filter_cap.m may be zero.
subcar1=(1+rolloff)/2+0.01e9/BaudRate; % relate to AWGBitRate and taps

%%生成信号
% rng(80);
DCOdata=randi(QAMorder,numofsymbols,1)-1;
% save(['./txdata/decimal',num2str(QAMorder),'.txt'],'DCOdata','-ascii');
% save(['./NN_2/decimal',num2str(QAMorder),'.txt'],'DCOdata','-ascii');
save(['D:/git_CTLE/CTLE_v2/data/decimal',num2str(QAMorder),'.txt'],'DCOdata','-ascii');