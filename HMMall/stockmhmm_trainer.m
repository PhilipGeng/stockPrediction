addpath(genpath('D:/Documents/MATLAB/HMMall'));

t=3; %window size, length of sequence
tr_portion=0.8; %portion of training set
%global settings
s=50; %num of states
m=10; %num of mixture components
o=1;

1
clear close;
load('stock11close.mat'); %load label to var label1
close11=close;
[errtrain11,errtest11,trvec11,testvec11,predicted11] = stock_mhmm(close11,t,tr_portion,s,m,o)

2
clear close;
load('stock23close.mat'); %load label to var label1
close23=close;
[errtrain23,errtest23,trvec23,testvec23,predicted23] = stock_mhmm(close23,t,tr_portion,s,m,o)

3
clear close;
load('stock1close.mat'); %load label to var label1
close1=close;
[errtrain1,errtest1,trvec1,testvec1,predicted1] = stock_mhmm(close1,t,tr_portion,s,m,o)

4
clear close;
load('stock13close.mat'); %load label to var label1
close13=close;
[errtrain13,errtest13,trvec13,testvec13,predicted13] = stock_mhmm(close13,t,tr_portion,s,m,o)

5
clear close;
load('stock293close.mat'); %load label to var label1
close293=close;
[errtrain293,errtest293,trvec293,testvec293,predicted293] = stock_mhmm(close293,t,tr_portion,s,m,o)

6
clear close;
load('stock857close.mat'); %load label to var label1
close857=close;
[errtrain857,errtest857,trvec857,testvec857,predicted857] = stock_mhmm(close857,t,tr_portion,s,m,o)