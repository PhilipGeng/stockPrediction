clear all;
addpath(genpath('D:/Documents/MATLAB/HMMall'));

t=5; %window size, length of sequence
tr_portion=0.8; %portion of training set
load('stk11label.mat'); %load label to var label1

%global settings
s=15; %num of states

label=label+1; %standardize labels
o=max(label);

train = label(1:length(label)*tr_portion);
test = label(length(label)*tr_portion-t:length(label));
%shufflezip
data = [];
for i=1:1:length(test)-t
    data = [data test(i:i+t-1)];
end
test=transpose(data);

data = [];
for i=1:1:length(train)-t
    data = [data train(i:i+t-1)];
end
testtrain=transpose(data);


%rand init
dp1=normalise(rand(s,1));
dtran1=mk_stochastic(rand(s,s));
dobs1=mk_stochastic(rand(s,o));

%randomly get sample data
%dsample = dhmm_sample(dp1,dtran1,dobs1,n,t);

%EM train
[LL, dp2, dtran2, dobs2] = dhmm_em(train, dp1, dtran1, dobs1, 'max_iter', 5);

%test
err=0;
predict=[];
for i=transpose(test)
   maxloglik = -10000;
   maxlabel = 1;
   testData = i;
   target = testData(t);
   for j=1:1:o
       testData(t)=j;
       loglik = dhmm_logprob(testData, dp2, dtran2, dobs2);
       if loglik>maxloglik
            maxlabel=j;
            maxloglik=loglik;
       end
   end
   errlocal = abs(target-maxlabel);
   err = err+errlocal;
   predict=[predict maxlabel];
end
acc_test = 1-err/length(test)

err=0;
predict=[];
for i=transpose(testtrain)
   maxloglik = -10000;
   maxlabel = 1;
   testData = i;
   target = testData(t);
   for j=1:1:o
       testData(t)=j;
       loglik = dhmm_logprob(testData, dp2, dtran2, dobs2);
       if loglik>maxloglik
            maxlabel=j;
            maxloglik=loglik;
       end
   end
   errlocal = abs(target-maxlabel);
   err = err+errlocal;
   predict=[predict maxlabel];
end
acc_train = 1-err/length(testtrain)
