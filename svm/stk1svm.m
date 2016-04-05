close all;
clear all;
load('stk857full.mat');

data = [WILLR,rocr3,rocr12,mom1,mom3,ema6,EMA12,MACD,obv,rsi6,rsi12,atr14,mfi14,cci12,cci20,trix];
data = [data label];
%pca on data
[pc,score,latent,tsquare] = princomp(data);

tranMatrix=pc(:,1:8);
pcad = data*(tranMatrix);
pcacum = cumsum(latent)./sum(latent);
pcavariance = pcacum(8)
data=[pcad label];
n = length(data(1,:));
[train,test]=randomSplit(data,0.8);

svmmodel = svmtrain(train(:,1:n-1),train(:,n),'kernel_function','mlp');
total=0;
right=0;
for i = 1:1:length(test)
    total = total + 1;%counter
    predlabel = svmclassify(svmmodel,test(i,1:n-1));
    if(test(i,n)==predlabel) %if the same
        right = right + 1;%counter
    end
end
total;
correct_rate = right/total