clear all;
load('data.mat')
addpath(genpath('D:/Documents/MATLAB/HMMall'));

t=10; %window size, length of sequence
tr_portion=0.8; %portion of training set

a=max(earn)+0.05;
b=min(earn)-0.05;
o=2;
slot = (a-b)/o;
for i=1:1:length(earn)
    earn(i)=ceil((earn(i)-b)/slot);
end

train = earn(1:floor(length(earn)*tr_portion));
test = earn(ceil(length(earn)*tr_portion):length(earn));
s=4;

transmat = mk_stochastic(rand(s,s));
obsmat = mk_stochastic(rand(s,o));
[estTR,estE] = hmmtrain(train,transmat,obsmat);
predicted = [];
err=0;
for i=t:1:length(test)
    seq=test(i-(t-1):i);
    reallabel=seq(t);
    classified = 1;
    loglik = -10000;
    seq=transpose(seq);
    for j=1:1:o
        seq(t)=j;
        [PSTATES,logpseq] = hmmdecode(seq,estTR,estE);
        if logpseq>loglik
            loglik=logpseq;
            classified=j;
        end
    end
    predicted = [predicted classified];
    err = err + abs(classified-reallabel);
end
