function [errtrain, errtest,err_trainvec,err_testvec,predicted] = stock_mhmm(close,t,tr_portion,s,m,o)
    raw = close;
    data=(transpose(raw(1:tr_portion*length(raw))));
    nex=floor(length(data)/t);
    nexfull = floor(length(raw)/t);
    data = data(1:nex*t);
    data = reshape(data,[o t nex]);

    mp0=normalise(rand(s,1));
    mtran0=mk_stochastic(rand(s,s));
    [mu0, Sigma0] = mixgauss_init(s*m, reshape(data,[o t*nex]), 'spherical');
    mu0 = reshape(mu0, [o s m]);
    Sigma0 = reshape(Sigma0, [o o s m]);
    mixmat0 = mk_stochastic(rand(s,m));
    [LL, prior1, transmat1, mu1, Sigma1, mixmat1] = ...
        mhmm_em(data, mp0, mtran0, mu0, Sigma0, mixmat0, 'max_iter', 25);

    a = min(raw);
    b = max(raw);
    discretizer = (b-a)/100;
    disseq = a:discretizer:b;

    target = [];
    predicted = [];
    for i=t:1:nexfull*t+t-1
        i
        test = raw(i-t+1:i);
        real = test(t);
        target=[target real];
        maxloglik = -10000;
        maxlab = 1;
        for j=1:1:length(disseq)
            cp = test;
            cp(t)=disseq(j);
            loglik = mhmm_logprob(reshape(cp,[o t 1]), prior1, transmat1, mu1, Sigma1, mixmat1);
            if loglik>maxloglik
                maxloglik=loglik;
                maxlab=disseq(j);
            end
        end
        predicted=[predicted maxlab];
    end
    errvec = predicted-target;
    err_trainvec = errvec(1:floor(tr_portion*length(errvec))+1);
    err_testvec = errvec(floor(tr_portion*length(errvec))+1:length(errvec));
    errtrain = sqrt(sumsqr(err_trainvec)/length(err_trainvec));
    errtest = sqrt(sumsqr(err_testvec)/length(err_testvec));
