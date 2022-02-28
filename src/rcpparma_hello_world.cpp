// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
#define ARMA_DONT_PRINT_ERRORS
// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

// simple example of creating two matrices and
// returning the result of an operatioon on them
//
// via the exports attribute we tell Rcpp to make this function
// available from R
//
// [[Rcpp::export]]
arma::mat rcpparma_hello_world() {
    arma::mat m1 = arma::eye<arma::mat>(3, 3);
    arma::mat m2 = arma::eye<arma::mat>(3, 3);
	                     
    return m1 + 3 * (m1 + m2);
}


// another simple example: outer product of a vector, 
// returning a matrix
//
// [[Rcpp::export]]
arma::mat rcpparma_outerproduct(const arma::colvec & x) {
    arma::mat m = x * x.t();
    return m;
}

// and the inner product returns a scalar
//
// [[Rcpp::export]]
double rcpparma_innerproduct(const arma::colvec & x) {
    double v = arma::as_scalar(x.t() * x);
    return v;
}


// and we can use Rcpp::List to return both at the same time
//
// [[Rcpp::export]]
Rcpp::List rcpparma_bothproducts(const arma::colvec & x) {
    arma::mat op = x * x.t();
    double    ip = arma::as_scalar(x.t() * x);
    return Rcpp::List::create(Rcpp::Named("outer")=op,
                              Rcpp::Named("inner")=ip);
}

// [[Rcpp::export]]
arma::vec elementwise_pow(const arma::vec&A,const arma::vec&p){
    int n=A.n_elem;
    arma::vec result=arma::zeros(n);
    
    for(int i=0;i<n;i++){
        result(i)=std::pow(A(i),p(i));
    }
    return result;
}

// [[Rcpp::export]]

arma::umat TmatL(const arma::mat&Inspec,const arma::vec&Timepoints){
    int n=Inspec.n_rows;
    int m=Timepoints.n_elem;
    arma::umat result(m,n);
    for(int i=0;i<n;i++){
        result.col(i)=Timepoints<=Inspec(i,0);
    }
    return result;
}
// [[Rcpp::export]]
arma::umat TmatR(const arma::mat&Inspec,const arma::vec&Timepoints){
    int n=Inspec.n_rows;
    int m=Timepoints.n_elem;
    arma::umat result(m,n);
    for(int i=0;i<n;i++){
        result.col(i)=Timepoints<=Inspec(i,1);
    }
    return result;
}

// [[Rcpp::export]]

arma::umat TmatLR(const arma::mat&Inspec,const arma::vec&Timepoints){
    int n=Inspec.n_rows;
    int m=Timepoints.n_elem;
    arma::umat result(m,n);
    for(int i=0;i<n;i++){
        result.col(i)=Timepoints<=Inspec(i,1)&&Timepoints>Inspec(i,0);
    }
    return result;
}


// [[Rcpp::export]]
double Log_likelihood(const arma::mat&X,const arma::mat&Delta,const arma::mat&mimic,const arma::umat&tL,
                      const arma::umat&tLR,const arma::umat&tR,const arma::vec&beta0,const arma::vec&lambda0){
    int betadim=X.n_cols;
    arma::mat L=mimic.col(0);
    arma::mat R=mimic.col(1);
    arma::mat XR=X%repmat(R,1,betadim);
    arma::mat XL=X%repmat(L,1,betadim);
    arma::mat XLR=XR-XL;
    arma::mat LambdaR=trans(trans(exp(lambda0))*tR);
    arma::mat betaXR=XR*beta0;
    arma::mat LambdaL=trans(trans(exp(lambda0))*tL);
    arma::mat betaXL=XL*beta0;
    arma::mat UL=LambdaL+betaXL;
    arma::mat UR=LambdaR+betaXR;
    arma::mat result=log(elementwise_pow((1-exp(-UL)),Delta.col(0)))+
        log(elementwise_pow(exp(-UL)-exp(-UR),Delta.col(1)))+
        log(elementwise_pow(exp(-UR),Delta.col(2)));
    return accu(result);
}



// [[Rcpp::export]]
arma::field<arma::mat> Update_cpp_Zero(const arma::mat&X,const arma::mat&Delta,const arma::mat&Inspec,const arma::mat&mimic,const arma::umat&tL,
                                       const arma::umat&tLR,const arma::umat&tR,const arma::mat&beta0,const arma::mat&lambda0){
    int betadim=X.n_cols;
    arma::mat tktemp=sort(unique(Inspec));
    arma::vec tk=tktemp.col(0);
    int n=Delta.n_rows;
    arma::vec onesvec=arma::ones(n);
    
    tk=tk.subvec(1,tk.n_elem-1);
    int m=tk.n_elem;
    arma::mat L=mimic.col(0);
    arma::mat R=mimic.col(1);
    arma::mat XR=X%repmat(R,1,betadim);
    arma::mat XL=X%repmat(L,1,betadim);
    arma::mat XLR=XR-XL;
    arma::mat LambdaR=trans(trans(exp(lambda0))*tR);
    arma::mat betaXR=XR*beta0;
    arma::mat LambdaL=trans(trans(exp(lambda0))*tL);
    arma::umat Lu=find(Delta.col(0) == 0);
    arma::umat LRu=find(Delta.col(1) == 0);
    
    arma::mat betaXL=XL*beta0;
    arma::mat betaXLR=XLR*beta0;
    arma::mat LambdaLR=trans(trans(exp(lambda0))*tLR);
    arma::mat UL=LambdaL+betaXL;
    arma::mat UR=LambdaR+betaXR;
    arma::mat ULR=LambdaLR+betaXLR;
    
    arma::vec onesvecm=arma::ones(m);
    
    arma::mat ULRindex=trans(trans(onesvecm)*tLR);
    arma::umat ULRu=find(ULRindex==0);
    ULR.elem(ULRu)=arma::ones(ULRu.n_elem,1);
    arma::mat A1L=exp(-UL)/(1-exp(-UL));
    arma::mat A2L=exp(-UL)/(2*pow(1-exp(-UL),2));
    A1L.elem( Lu )=arma::zeros(Lu.n_elem,1);
    A2L.elem( Lu )=arma::zeros(Lu.n_elem,1);
    // std::cout<<UL;
    arma::mat A3L=A1L+2*A2L%UL;
    
    
    arma::mat A1LR=exp(-ULR)/(1-exp(-ULR));
    arma::mat A2LR=exp(-ULR)/(2*pow(1-exp(-ULR),2));
    A1LR.elem( LRu )=arma::zeros(LRu.n_elem,1);
    A2LR.elem( LRu )=arma::zeros(LRu.n_elem,1);
    
    arma::mat A3LR=A1LR+2*A2LR%ULR;
    
    // std::cout<<A3LR;
    arma::mat Lmat=repmat(trans(Delta.col(0)),m,1);
    arma::mat Imat=repmat(trans(Delta.col(1)),m,1);
    arma::mat Rmat=repmat(trans(Delta.col(2)),m,1);
    arma::mat ULmat=repmat(trans(UL),m,1);
    arma::mat A2Lmat=repmat(trans(A2L),m,1);
    arma::mat A3Lmat=repmat(trans(A3L),m,1);
    arma::mat ULRmat=repmat(trans(ULR),m,1);
    arma::mat A2LRmat=repmat(trans(A2LR),m,1);
    arma::mat A3LRmat=repmat(trans(A3LR),m,1);
    arma::mat lambda0mat=repmat(exp(lambda0),1,n);
    arma::mat FirstDerivMat=(Lmat%tL%(A3Lmat-2*A2Lmat%ULmat)-
        Imat%tL-Rmat%tR+Imat%tLR%(A3LRmat-2*A2LRmat%ULRmat))%lambda0mat;
    arma::mat SecondDerivMat=(Lmat%tL%(A3Lmat-4*A2Lmat%ULmat-2/ULmat)-
        Imat%tL-Rmat%tR+Imat%tLR%(A3LRmat-4*A2LRmat%ULRmat-2/ULRmat))%lambda0mat;
    arma::mat FirstDeriv=FirstDerivMat*onesvec;
    arma::mat SecondDeriv=SecondDerivMat*onesvec;
    // std::cout<<ULR;
    arma::mat lambdanew=lambda0-FirstDeriv/SecondDeriv;
    
    arma::umat Lambdau=find(FirstDeriv.col(0) == 0&&SecondDeriv.col(0) == 0);
    // std::cout<<lambdanew;
    lambdanew(Lambdau)=arma::ones(Lambdau.n_elem,1)*(-200);
    arma::mat BetaFirst=trans(Delta.col(0)/UL)*XL+trans(Delta.col(1)/ULR)*XLR+
        trans(Delta.col(0)%(A3L-1/UL))*XL-trans(Delta.col(1))*XL-trans(Delta.col(2))*XR+
        trans((Delta.col(1)%(A3LR-1/ULR)))*XLR-
        2*(trans(Delta.col(0)%A2L%UL)*XL+trans(Delta.col(1)%A2LR%ULR)*XLR);
    // std::cout<<(A2L.row(58));
    arma::mat BetaSecond=arma::zeros(betadim,betadim);
    for(int i=0;i<n;i++){
        if(X.row(i).is_zero()){
            BetaSecond=BetaSecond+arma::zeros(betadim,betadim);
        }
        else{
            BetaSecond=BetaSecond+(-2*Delta(i,0)/betaXL(i,0)/UL(i,0)-2*Delta(i,0)*A2L(i,0)*UL(i,0)/betaXL(i,0))*trans(XL.row(i))*XL.row(i)+
                (-2*Delta(i,1)/betaXLR(i,0)/ULR(i,0)-2*Delta(i,1)*A2LR(i,0)*ULR(i,0)/betaXLR(i,0))*trans(XLR.row(i))*XLR.row(i);
        }
        
    }
    // std::cout<<BetaFirst;
    arma::mat betanew=beta0-solve(BetaSecond,trans(BetaFirst));
    // std::cout<<betanew;
    arma::field<arma::mat> result(2);
    result(0)=lambdanew;
    result(1)=betanew;
    return result;
}





// [[Rcpp::export]]
arma::field<arma::mat> Main_func_cpp(const arma::mat&X,const arma::mat&Delta,const arma::mat&Inspec,const arma::mat&mimic,const int&Max_iter,
                                     const double&Tol){
    int betadim=X.n_cols;
    arma::mat tktemp=sort(unique(Inspec));
    arma::vec tk=tktemp.col(0);
    int n=Delta.n_rows;
    arma::vec onesvec=arma::ones(n);
    tk=tk.subvec(1,tk.n_elem-1);
    int m=tk.n_elem;
    arma::mat L=mimic.col(0);
    arma::mat R=mimic.col(1);
    arma::umat tL=TmatL(mimic,tk);
    arma::umat tR=TmatR(mimic,tk);
    arma::umat tLR=TmatLR(mimic,tk);
    arma::mat beta0=arma::ones(betadim,1)*0.5;
    arma::mat lambda0=arma::ones(m,1)*(-11);
    double log_likelihood_initial=Log_likelihood(X,Delta,mimic,tL,tLR,tR,beta0,lambda0);
    arma::mat log_likelihood_list=arma::ones(1,1)*log_likelihood_initial;
    arma::field<arma::mat> newresult(2);
    int iter=0;
    double difference=20000;
    double absdiff=1000;
    arma::mat differencelist=arma::ones(1,1)*difference;
    arma::mat lambdaold=lambda0;
    arma::mat betaold=beta0;
    arma::mat absdifflist=arma::ones(1,1)*absdiff;
    do{
        newresult=Update_cpp_Zero(X,Delta,Inspec,mimic,tL,tLR,tR,beta0,lambda0);
        lambda0=newresult(0);
        beta0=newresult(1);
        absdiff=accu(abs(exp(lambda0)-exp(lambdaold)))+accu(abs(beta0-betaold));
        lambdaold=lambda0;
        betaold=beta0;
        double log_likelihood=Log_likelihood(X,Delta,mimic,tL,tLR,tR,beta0,lambda0);
        log_likelihood_list=join_cols(log_likelihood_list,arma::ones(1,1)*log_likelihood);
        iter=iter+1;
        difference=(log_likelihood_list(iter,0)-log_likelihood_list(iter-1,0))/std::abs(log_likelihood_list(iter-1,0));
        differencelist=join_cols(differencelist,arma::ones(1,1)*difference);
        absdifflist=join_cols(absdifflist,arma::ones(1,1)*absdiff);
    } while (iter<100||(iter<Max_iter&&absdiff>Tol));
    arma::field<arma::mat> finalresult(3);
    finalresult(0)=newresult(1);
    finalresult(1)=newresult(0);
    finalresult(2)=log_likelihood_list(iter,0);
    // finalresult(3)=absdifflist;
    return finalresult;
}



// [[Rcpp::export]]
arma::mat Profile_Update_cpp(const arma::mat&X,const arma::mat&Delta,const arma::mat&Inspec,const arma::mat&mimic,const arma::umat&tL,
                             const arma::umat&tLR,const arma::umat&tR,const arma::mat&beta0,const arma::mat&lambda0){
    int betadim=X.n_cols;
    arma::mat tktemp=sort(unique(Inspec));
    arma::vec tk=tktemp.col(0);
    int n=Delta.n_rows;
    arma::vec onesvec=arma::ones(n);
    tk=tk.subvec(1,tk.n_elem-1);
    int m=tk.n_elem;
    arma::mat L=mimic.col(0);
    arma::mat R=mimic.col(1);
    arma::mat XR=X%repmat(R,1,betadim);
    arma::mat XL=X%repmat(L,1,betadim);
    arma::mat XLR=XR-XL;
    arma::mat LambdaR=trans(trans(exp(lambda0))*tR);
    arma::mat betaXR=XR*beta0;
    arma::mat LambdaL=trans(trans(exp(lambda0))*tL);
    arma::umat Lu=find(Delta.col(0) == 0);
    arma::umat LRu=find(Delta.col(1) == 0);
    
    arma::mat betaXL=XL*beta0;
    arma::mat betaXLR=XLR*beta0;
    arma::mat LambdaLR=trans(trans(exp(lambda0))*tLR);
    arma::mat UL=LambdaL+betaXL;
    arma::mat UR=LambdaR+betaXR;
    arma::mat ULR=LambdaLR+betaXLR;
    arma::vec onesvecm=arma::ones(m);
    
    arma::mat ULRindex=trans(trans(onesvecm)*tLR);
    arma::umat ULRu=find(ULRindex==0);
    ULR.elem(ULRu)=arma::ones(ULRu.n_elem,1);
    
    
    arma::mat A1L=exp(-UL)/(1-exp(-UL));
    arma::mat A2L=exp(-UL)/(2*pow(1-exp(-UL),2));
    A1L.elem( Lu )=arma::zeros(Lu.n_elem,1);
    A2L.elem( Lu )=arma::zeros(Lu.n_elem,1);
    // std::cout<<A2L.row(58);
    arma::mat A3L=A1L+2*A2L%UL;
    
    
    arma::mat A1LR=exp(-ULR)/(1-exp(-ULR));
    arma::mat A2LR=exp(-ULR)/(2*pow(1-exp(-ULR),2));
    A1LR.elem( LRu )=arma::zeros(LRu.n_elem,1);
    A2LR.elem( LRu )=arma::zeros(LRu.n_elem,1);
    
    arma::mat A3LR=A1LR+2*A2LR%ULR;
    // std::cout<<UL;
    
    arma::mat Lmat=repmat(trans(Delta.col(0)),m,1);
    arma::mat Imat=repmat(trans(Delta.col(1)),m,1);
    arma::mat Rmat=repmat(trans(Delta.col(2)),m,1);
    arma::mat ULmat=repmat(trans(UL),m,1);
    arma::mat A2Lmat=repmat(trans(A2L),m,1);
    arma::mat A3Lmat=repmat(trans(A3L),m,1);
    arma::mat ULRmat=repmat(trans(ULR),m,1);
    arma::mat A2LRmat=repmat(trans(A2LR),m,1);
    arma::mat A3LRmat=repmat(trans(A3LR),m,1);
    arma::mat lambda0mat=repmat(exp(lambda0),1,n);
    arma::mat FirstDerivMat=(Lmat%tL%(A3Lmat-2*A2Lmat%ULmat)-
        Imat%tL-Rmat%tR+Imat%tLR%(A3LRmat-2*A2LRmat%ULRmat))%lambda0mat;
    arma::mat SecondDerivMat=(Lmat%tL%(A3Lmat-4*A2Lmat%ULmat-2/ULmat)-
        Imat%tL-Rmat%tR+Imat%tLR%(A3LRmat-4*A2LRmat%ULRmat-2/ULRmat))%lambda0mat;
    arma::mat FirstDeriv=FirstDerivMat*onesvec;
    arma::mat SecondDeriv=SecondDerivMat*onesvec;
    arma::mat lambdanew=lambda0-FirstDeriv/SecondDeriv;
    arma::umat Lambdau=find(FirstDeriv.col(0) == 0&&SecondDeriv.col(0) == 0);
    lambdanew(Lambdau)=arma::ones(Lambdau.n_elem,1)*(-200);
    
    return lambdanew;
}

// [[Rcpp::export]]
double Profile_Main_cpp(const arma::mat&beta0,const arma::mat&lambda0,const arma::mat&X,const arma::mat&Delta,const arma::mat&Inspec,const arma::mat&mimic,const int&Max_iter,
                        const double&Tol){

    // std::cout<<betadim;
    arma::mat tktemp=sort(unique(Inspec));
    arma::vec tk=tktemp.col(0);
    int n=Delta.n_rows;
    arma::vec onesvec=arma::ones(n);
    tk=tk.subvec(1,tk.n_elem-1);
    arma::mat L=mimic.col(0);
    arma::mat R=mimic.col(1);
    arma::umat tL=TmatL(mimic,tk);
    arma::umat tR=TmatR(mimic,tk);
    arma::umat tLR=TmatLR(mimic,tk);
    double log_likelihood_initial=Log_likelihood(X,Delta,mimic,tL,tLR,tR,beta0,lambda0);
    // std::cout<<log_likelihood_initial;
    arma::mat log_likelihood_list=arma::ones(1,1)*log_likelihood_initial;
    arma::mat lambdanew;
    int iter=0;
    double difference=20000;
    double absdiff=1000;
    arma::mat differencelist=arma::ones(1,1)*difference;
    arma::mat lambdaold=lambda0;
    arma::mat absdifflist=arma::ones(1,1)*absdiff;
    do{
        lambdanew=Profile_Update_cpp(X,Delta,Inspec,mimic,tL,tLR,tR,beta0,lambdaold);
        
        absdiff=accu(abs(exp(lambdanew)-exp(lambdaold)));
        lambdaold=lambdanew;
        double log_likelihood=Log_likelihood(X,Delta,mimic,tL,tLR,tR,beta0,lambdanew);
        log_likelihood_list=join_cols(log_likelihood_list,arma::ones(1,1)*log_likelihood);
        iter=iter+1;
        // std::cout<<iter;
        difference=(log_likelihood_list(iter,0)-log_likelihood_list(iter-1,0))/std::abs(log_likelihood_list(iter-1,0));
        differencelist=join_cols(differencelist,arma::ones(1,1)*difference);
        absdifflist=join_cols(absdifflist,arma::ones(1,1)*absdiff);
    } while (iter<Max_iter&&absdiff>Tol);
    
    return log_likelihood_list(iter,0);
}



// [[Rcpp::export]]

arma::mat SE_cal_cpp_one(const arma::mat&beta0,const arma::mat&lambda0,const double&lgvalue,const arma::mat&X,const arma::mat&Delta,const arma::mat&Inspec,const arma::mat&mimic,const int&Max_iter,
                         const double&Tol,const double&hn){
    int betadim=X.n_cols;
    arma::mat e10=arma::zeros(betadim,1);
    e10(0,0)=1;
    arma::mat beta10=beta0+hn*e10;
    double lg10=Profile_Main_cpp(beta10,lambda0,X,Delta,Inspec,mimic,Max_iter,Tol);
    arma::mat beta1010=beta0+hn*e10+hn*e10;
    double lg1010=Profile_Main_cpp(beta1010,lambda0,X,Delta,Inspec,mimic,Max_iter,Tol);
    arma::mat InfoMat=arma::zeros(betadim,betadim);
    InfoMat(0,0)=lgvalue-2*lg10+lg1010;
    InfoMat=-InfoMat/std::pow(hn,2);
    arma::mat CovMat=InfoMat.i();
    // std::cout<<lg10;
    
    arma::mat se=sqrt(CovMat.diag());
    return se;
}





// [[Rcpp::export]]

arma::mat SE_cal_cpp(const arma::mat&beta0,const arma::mat&lambda0,const double&lgvalue,const arma::mat&X,const arma::mat&Delta,const arma::mat&Inspec,const arma::mat&mimic,const int&Max_iter,
                     const double&Tol,const double&hn){
    int betadim=X.n_cols;
    arma::mat e10=arma::zeros(betadim,1);
    arma::mat e01=arma::zeros(betadim,1);
    e10(0,0)=1;
    e01(1,0)=1;
    arma::mat beta10=beta0+hn*e10;
    double lg10=Profile_Main_cpp(beta10,lambda0,X,Delta,Inspec,mimic,Max_iter,Tol);
    arma::mat beta1010=beta0+hn*e10+hn*e10;
    double lg1010=Profile_Main_cpp(beta1010,lambda0,X,Delta,Inspec,mimic,Max_iter,Tol);
    arma::mat beta01=beta0+hn*e01;
    double lg01=Profile_Main_cpp(beta01,lambda0,X,Delta,Inspec,mimic,Max_iter,Tol);
    arma::mat beta0101=beta0+hn*e01+hn*e01;
    double lg0101=Profile_Main_cpp(beta0101,lambda0,X,Delta,Inspec,mimic,Max_iter,Tol);
    arma::mat beta1001=beta0+hn*e10+hn*e01;
    double lg1001=Profile_Main_cpp(beta1001,lambda0,X,Delta,Inspec,mimic,Max_iter,Tol);
    arma::mat InfoMat=arma::zeros(2,2);
    InfoMat(0,0)=lgvalue-2*lg10+lg1010;
    InfoMat(1,1)=lgvalue-2*lg01+lg0101;
    InfoMat(1,0)=InfoMat(0,1)=lgvalue-lg10-lg01+lg1001;
    InfoMat=-InfoMat/std::pow(hn,2);
    arma::mat CovMat=InfoMat.i();
    arma::mat se=sqrt(CovMat.diag());
    return se;
}


// [[Rcpp::export]]
arma::mat SE_cal_cpp_gen(const arma::mat&beta0,const arma::mat&lambda0,const double&lgvalue,const arma::mat&X,const arma::mat&Delta,const arma::mat&Inspec,const arma::mat&mimic,const int&Max_iter,
                     const double&Tol,const double&hn){
    int betadim=X.n_cols;
    arma::mat E;
    E.eye(betadim,betadim);
    arma::mat InfoMat;
    InfoMat.zeros(betadim,betadim);
    for(int r=0;r<betadim;r++){
        for(int s=r;s<betadim;s++){
            arma::mat betar=beta0+hn*E.col(r);
            arma::mat betas=beta0+hn*E.col(s);
            arma::mat betars=beta0+hn*E.col(r)+hn*E.col(s);
            double lgr=Profile_Main_cpp(betar,lambda0,X,Delta,Inspec,mimic,Max_iter,Tol);
            double lgs=Profile_Main_cpp(betas,lambda0,X,Delta,Inspec,mimic,Max_iter,Tol);
            double lgrs=Profile_Main_cpp(betars,lambda0,X,Delta,Inspec,mimic,Max_iter,Tol);
            InfoMat(r,s)=lgvalue-lgr-lgs+lgrs;
        }
    }
    InfoMat=InfoMat+InfoMat.t();
    InfoMat.diag()=InfoMat.diag()/2;
    InfoMat=-InfoMat/std::pow(hn,2);
    arma::mat CovMat=InfoMat.i();
    arma::mat se=sqrt(CovMat.diag());
    return se;
}


// [[Rcpp::export]]
double Log_likelihood_test(const arma::vec&parameter,const arma::mat&X,const arma::mat&Delta,const arma::mat&mimic,const arma::umat&tL,
                           const arma::umat&tLR,const arma::umat&tR){
    int betadim=X.n_cols;
    
    arma::mat beta0=parameter.subvec(0,betadim-1);
    arma::mat lambda0=parameter.subvec(betadim,parameter.n_rows-1);
    arma::mat L=mimic.col(0);
    arma::mat R=mimic.col(1);
    // std::cout<<L;
    arma::mat XR=X%R;
    arma::mat XL=X%L;
    // std::cout<<XR;
    arma::mat XLR=XR-XL;
    arma::mat LambdaR=trans(trans(exp(lambda0))*tR);
    // std::cout<<beta0;
    
    arma::mat betaXR=XR*beta0;
    arma::mat LambdaL=trans(trans(exp(lambda0))*tL);
    arma::mat betaXL=XL*beta0;
    arma::mat UL=LambdaL+betaXL;
    arma::mat UR=LambdaR+betaXR;
    arma::mat result=log(elementwise_pow((1-exp(-UL)),Delta.col(0)))+
        log(elementwise_pow(exp(-UL)-exp(-UR),Delta.col(1)))+
        log(elementwise_pow(exp(-UR),Delta.col(2)));
    return -accu(result);
}



// [[Rcpp::export]]
double Log_likelihood_Dep2(const arma::mat&X,const arma::mat&Delta,const arma::mat&mimic,const arma::umat&tL,
                           const arma::umat&tLR,const arma::umat&tR,const arma::vec&beta0,const arma::vec&lambda0){
    int betadim=X.n_cols;
    arma::mat L=mimic.col(0);
    arma::mat R=mimic.col(1);
    arma::mat XR=X%repmat(exp(R)-1,1,betadim);
    arma::mat XL=X%repmat(exp(L)-1,1,betadim);
    arma::mat XLR=XR-XL;
    arma::mat LambdaR=trans(trans(exp(lambda0))*tR);
    arma::mat betaXR=XR*beta0;
    arma::mat LambdaL=trans(trans(exp(lambda0))*tL);
    arma::mat betaXL=XL*beta0;
    arma::mat UL=LambdaL+betaXL;
    arma::mat UR=LambdaR+betaXR;
    arma::mat result=log(elementwise_pow((1-exp(-UL)),Delta.col(0)))+
        log(elementwise_pow(exp(-UL)-exp(-UR),Delta.col(1)))+
        log(elementwise_pow(exp(-UR),Delta.col(2)));
    return accu(result);
}



// [[Rcpp::export]]
arma::field<arma::mat> Update_cpp_Dep2(const arma::mat&X,const arma::mat&Delta,const arma::mat&Inspec,const arma::mat&mimic,const arma::umat&tL,
                                       const arma::umat&tLR,const arma::umat&tR,const arma::mat&beta0,const arma::mat&lambda0){
    int betadim=X.n_cols;
    arma::mat tktemp=sort(unique(Inspec));
    arma::vec tk=tktemp.col(0);
    int n=Delta.n_rows;
    arma::vec onesvec=arma::ones(n);
    
    tk=tk.subvec(1,tk.n_elem-1);
    int m=tk.n_elem;
    arma::mat L=mimic.col(0);
    arma::mat R=mimic.col(1);
    arma::mat XR=X%repmat(exp(R)-1,1,betadim);
    arma::mat XL=X%repmat(exp(L)-1,1,betadim);
    
    arma::mat XLR=XR-XL;
    arma::mat LambdaR=trans(trans(exp(lambda0))*tR);
    arma::mat betaXR=XR*beta0;
    arma::mat LambdaL=trans(trans(exp(lambda0))*tL);
    arma::umat Lu=find(Delta.col(0) == 0);
    arma::umat LRu=find(Delta.col(1) == 0);
    
    arma::mat betaXL=XL*beta0;
    arma::mat betaXLR=XLR*beta0;
    arma::mat LambdaLR=trans(trans(exp(lambda0))*tLR);
    arma::mat UL=LambdaL+betaXL;
    arma::mat UR=LambdaR+betaXR;
    arma::mat ULR=LambdaLR+betaXLR;
    
    arma::vec onesvecm=arma::ones(m);
    
    arma::mat ULRindex=trans(trans(onesvecm)*tLR);
    arma::umat ULRu=find(ULRindex==0);
    ULR.elem(ULRu)=arma::ones(ULRu.n_elem,1);
    arma::mat A1L=exp(-UL)/(1-exp(-UL));
    arma::mat A2L=exp(-UL)/(2*pow(1-exp(-UL),2));
    A1L.elem( Lu )=arma::zeros(Lu.n_elem,1);
    A2L.elem( Lu )=arma::zeros(Lu.n_elem,1);
    // std::cout<<UL;
    arma::mat A3L=A1L+2*A2L%UL;
    
    
    arma::mat A1LR=exp(-ULR)/(1-exp(-ULR));
    arma::mat A2LR=exp(-ULR)/(2*pow(1-exp(-ULR),2));
    A1LR.elem( LRu )=arma::zeros(LRu.n_elem,1);
    A2LR.elem( LRu )=arma::zeros(LRu.n_elem,1);
    
    arma::mat A3LR=A1LR+2*A2LR%ULR;
    
    // std::cout<<A3LR;
    arma::mat Lmat=repmat(trans(Delta.col(0)),m,1);
    arma::mat Imat=repmat(trans(Delta.col(1)),m,1);
    arma::mat Rmat=repmat(trans(Delta.col(2)),m,1);
    arma::mat ULmat=repmat(trans(UL),m,1);
    arma::mat A2Lmat=repmat(trans(A2L),m,1);
    arma::mat A3Lmat=repmat(trans(A3L),m,1);
    arma::mat ULRmat=repmat(trans(ULR),m,1);
    arma::mat A2LRmat=repmat(trans(A2LR),m,1);
    arma::mat A3LRmat=repmat(trans(A3LR),m,1);
    arma::mat lambda0mat=repmat(exp(lambda0),1,n);
    arma::mat FirstDerivMat=(Lmat%tL%(A3Lmat-2*A2Lmat%ULmat)-
        Imat%tL-Rmat%tR+Imat%tLR%(A3LRmat-2*A2LRmat%ULRmat))%lambda0mat;
    arma::mat SecondDerivMat=(Lmat%tL%(A3Lmat-4*A2Lmat%ULmat-2/ULmat)-
        Imat%tL-Rmat%tR+Imat%tLR%(A3LRmat-4*A2LRmat%ULRmat-2/ULRmat))%lambda0mat;
    arma::mat FirstDeriv=FirstDerivMat*onesvec;
    arma::mat SecondDeriv=SecondDerivMat*onesvec;
    // std::cout<<ULR;
    arma::mat lambdanew=lambda0-FirstDeriv/SecondDeriv;
    
    arma::umat Lambdau=find(FirstDeriv.col(0) == 0&&SecondDeriv.col(0) == 0);
    // std::cout<<lambdanew;
    lambdanew(Lambdau)=arma::ones(Lambdau.n_elem,1)*(-200);
    arma::mat BetaFirst=trans(Delta.col(0)/UL)*XL+trans(Delta.col(1)/ULR)*XLR+
        trans(Delta.col(0)%(A3L-1/UL))*XL-trans(Delta.col(1))*XL-trans(Delta.col(2))*XR+
        trans((Delta.col(1)%(A3LR-1/ULR)))*XLR-
        2*(trans(Delta.col(0)%A2L%UL)*XL+trans(Delta.col(1)%A2LR%ULR)*XLR);
    // std::cout<<(A2L.row(58));
    arma::mat BetaSecond=arma::zeros(betadim,betadim);
    for(int i=0;i<n;i++){
        if(X.row(i).is_zero()){
            BetaSecond=BetaSecond+arma::zeros(betadim,betadim);
        }
        else{
            BetaSecond=BetaSecond+(-2*Delta(i,0)/betaXL(i,0)/UL(i,0)-2*Delta(i,0)*A2L(i,0)*UL(i,0)/betaXL(i,0))*trans(XL.row(i))*XL.row(i)+
                (-2*Delta(i,1)/betaXLR(i,0)/ULR(i,0)-2*Delta(i,1)*A2LR(i,0)*ULR(i,0)/betaXLR(i,0))*trans(XLR.row(i))*XLR.row(i);
        }
        
    }
    // std::cout<<BetaFirst;
    arma::mat betanew=beta0-solve(BetaSecond,trans(BetaFirst));
    // std::cout<<betanew;
    arma::field<arma::mat> result(2);
    result(0)=lambdanew;
    result(1)=betanew;
    return result;
}



// [[Rcpp::export]]
arma::field<arma::mat> Main_func_Dep(const arma::mat&X,const arma::mat&Delta,const arma::mat&Inspec,const arma::mat&mimic,const int&Max_iter,
                                      const double&Tol){
    int betadim=X.n_cols;
    arma::mat tktemp=sort(unique(Inspec));
    arma::vec tk=tktemp.col(0);
    int n=Delta.n_rows;
    arma::vec onesvec=arma::ones(n);
    tk=tk.subvec(1,tk.n_elem-1);
    int m=tk.n_elem;
    arma::mat L=mimic.col(0);
    arma::mat R=mimic.col(1);
    arma::umat tL=TmatL(mimic,tk);
    arma::umat tR=TmatR(mimic,tk);
    arma::umat tLR=TmatLR(mimic,tk);
    arma::mat beta0=arma::ones(betadim,1)*0.5;
    arma::mat lambda0=arma::ones(m,1)*(-11);
    double log_likelihood_initial=Log_likelihood_Dep2(X,Delta,mimic,tL,tLR,tR,beta0,lambda0);
    arma::mat log_likelihood_list=arma::ones(1,1)*log_likelihood_initial;
    arma::field<arma::mat> newresult(2);
    int iter=0;
    double difference=20000;
    double absdiff=1000;
    arma::mat differencelist=arma::ones(1,1)*difference;
    arma::mat lambdaold=lambda0;
    arma::mat betaold=beta0;
    arma::mat absdifflist=arma::ones(1,1)*absdiff;
    do{
        newresult=Update_cpp_Dep2(X,Delta,Inspec,mimic,tL,tLR,tR,beta0,lambda0);
        lambda0=newresult(0);
        beta0=newresult(1);
        absdiff=accu(abs(exp(lambda0)-exp(lambdaold)))+accu(abs(beta0-betaold));
        lambdaold=lambda0;
        betaold=beta0;
        double log_likelihood=Log_likelihood_Dep2(X,Delta,mimic,tL,tLR,tR,beta0,lambda0);
        log_likelihood_list=join_cols(log_likelihood_list,arma::ones(1,1)*log_likelihood);
        iter=iter+1;
        difference=(log_likelihood_list(iter,0)-log_likelihood_list(iter-1,0))/std::abs(log_likelihood_list(iter-1,0));
        differencelist=join_cols(differencelist,arma::ones(1,1)*difference);
        absdifflist=join_cols(absdifflist,arma::ones(1,1)*absdiff);
    } while (iter<100||(iter<Max_iter&&absdiff>Tol));
    arma::field<arma::mat> finalresult(4);
    finalresult(0)=newresult(1);
    finalresult(1)=newresult(0);
    finalresult(2)=log_likelihood_list(iter,0);
    finalresult(3)=absdifflist;
    return finalresult;
}


// [[Rcpp::export]]
arma::mat Profile_Update_Dep2(const arma::mat&X,const arma::mat&Delta,const arma::mat&Inspec,const arma::mat&mimic,const arma::umat&tL,
                              const arma::umat&tLR,const arma::umat&tR,const arma::mat&beta0,const arma::mat&lambda0){
    int betadim=X.n_cols;
    arma::mat tktemp=sort(unique(Inspec));
    arma::vec tk=tktemp.col(0);
    int n=Delta.n_rows;
    arma::vec onesvec=arma::ones(n);
    tk=tk.subvec(1,tk.n_elem-1);
    int m=tk.n_elem;
    arma::mat L=mimic.col(0);
    arma::mat R=mimic.col(1);
    arma::mat XR=X%repmat(exp(R)-1,1,betadim);
    arma::mat XL=X%repmat(exp(L)-1,1,betadim);
    arma::mat XLR=XR-XL;
    arma::mat LambdaR=trans(trans(exp(lambda0))*tR);
    arma::mat betaXR=XR*beta0;
    arma::mat LambdaL=trans(trans(exp(lambda0))*tL);
    arma::umat Lu=find(Delta.col(0) == 0);
    arma::umat LRu=find(Delta.col(1) == 0);
    
    arma::mat betaXL=XL*beta0;
    arma::mat betaXLR=XLR*beta0;
    arma::mat LambdaLR=trans(trans(exp(lambda0))*tLR);
    arma::mat UL=LambdaL+betaXL;
    arma::mat UR=LambdaR+betaXR;
    arma::mat ULR=LambdaLR+betaXLR;
    arma::vec onesvecm=arma::ones(m);
    
    arma::mat ULRindex=trans(trans(onesvecm)*tLR);
    arma::umat ULRu=find(ULRindex==0);
    ULR.elem(ULRu)=arma::ones(ULRu.n_elem,1);
    
    
    arma::mat A1L=exp(-UL)/(1-exp(-UL));
    arma::mat A2L=exp(-UL)/(2*pow(1-exp(-UL),2));
    A1L.elem( Lu )=arma::zeros(Lu.n_elem,1);
    A2L.elem( Lu )=arma::zeros(Lu.n_elem,1);
    // std::cout<<A2L.row(58);
    arma::mat A3L=A1L+2*A2L%UL;
    
    
    arma::mat A1LR=exp(-ULR)/(1-exp(-ULR));
    arma::mat A2LR=exp(-ULR)/(2*pow(1-exp(-ULR),2));
    A1LR.elem( LRu )=arma::zeros(LRu.n_elem,1);
    A2LR.elem( LRu )=arma::zeros(LRu.n_elem,1);
    
    arma::mat A3LR=A1LR+2*A2LR%ULR;
    // std::cout<<UL;
    
    arma::mat Lmat=repmat(trans(Delta.col(0)),m,1);
    arma::mat Imat=repmat(trans(Delta.col(1)),m,1);
    arma::mat Rmat=repmat(trans(Delta.col(2)),m,1);
    arma::mat ULmat=repmat(trans(UL),m,1);
    arma::mat A2Lmat=repmat(trans(A2L),m,1);
    arma::mat A3Lmat=repmat(trans(A3L),m,1);
    arma::mat ULRmat=repmat(trans(ULR),m,1);
    arma::mat A2LRmat=repmat(trans(A2LR),m,1);
    arma::mat A3LRmat=repmat(trans(A3LR),m,1);
    arma::mat lambda0mat=repmat(exp(lambda0),1,n);
    arma::mat FirstDerivMat=(Lmat%tL%(A3Lmat-2*A2Lmat%ULmat)-
        Imat%tL-Rmat%tR+Imat%tLR%(A3LRmat-2*A2LRmat%ULRmat))%lambda0mat;
    arma::mat SecondDerivMat=(Lmat%tL%(A3Lmat-4*A2Lmat%ULmat-2/ULmat)-
        Imat%tL-Rmat%tR+Imat%tLR%(A3LRmat-4*A2LRmat%ULRmat-2/ULRmat))%lambda0mat;
    arma::mat FirstDeriv=FirstDerivMat*onesvec;
    arma::mat SecondDeriv=SecondDerivMat*onesvec;
    arma::mat lambdanew=lambda0-FirstDeriv/SecondDeriv;
    arma::umat Lambdau=find(FirstDeriv.col(0) == 0&&SecondDeriv.col(0) == 0);
    lambdanew(Lambdau)=arma::ones(Lambdau.n_elem,1)*(-200);
    
    return lambdanew;
}

// [[Rcpp::export]]
double Profile_Main_Dep2(const arma::mat&beta0,const arma::mat&lambda0,const arma::mat&X,const arma::mat&Delta,const arma::mat&Inspec,const arma::mat&mimic,const int&Max_iter,
                         const double&Tol){
    // std::cout<<betadim;
    arma::mat tktemp=sort(unique(Inspec));
    arma::vec tk=tktemp.col(0);
    int n=Delta.n_rows;
    arma::vec onesvec=arma::ones(n);
    tk=tk.subvec(1,tk.n_elem-1);
    arma::mat L=mimic.col(0);
    arma::mat R=mimic.col(1);
    arma::umat tL=TmatL(mimic,tk);
    arma::umat tR=TmatR(mimic,tk);
    arma::umat tLR=TmatLR(mimic,tk);
    double log_likelihood_initial=Log_likelihood_Dep2(X,Delta,mimic,tL,tLR,tR,beta0,lambda0);
    // std::cout<<log_likelihood_initial;
    arma::mat log_likelihood_list=arma::ones(1,1)*log_likelihood_initial;
    arma::mat lambdanew;
    int iter=0;
    double difference=20000;
    double absdiff=1000;
    arma::mat differencelist=arma::ones(1,1)*difference;
    arma::mat lambdaold=lambda0;
    arma::mat absdifflist=arma::ones(1,1)*absdiff;
    do{
        lambdanew=Profile_Update_Dep2(X,Delta,Inspec,mimic,tL,tLR,tR,beta0,lambdaold);
        
        absdiff=accu(abs(exp(lambdanew)-exp(lambdaold)));
        lambdaold=lambdanew;
        double log_likelihood=Log_likelihood_Dep2(X,Delta,mimic,tL,tLR,tR,beta0,lambdanew);
        log_likelihood_list=join_cols(log_likelihood_list,arma::ones(1,1)*log_likelihood);
        iter=iter+1;
        // std::cout<<iter;
        difference=(log_likelihood_list(iter,0)-log_likelihood_list(iter-1,0))/std::abs(log_likelihood_list(iter-1,0));
        differencelist=join_cols(differencelist,arma::ones(1,1)*difference);
        absdifflist=join_cols(absdifflist,arma::ones(1,1)*absdiff);
    } while (iter<Max_iter&&absdiff>Tol);
    
    return log_likelihood_list(iter,0);
}



// [[Rcpp::export]]

arma::mat SE_cal_Dep_one(const arma::mat&beta0,const arma::mat&lambda0,const double&lgvalue,const arma::mat&X,const arma::mat&Delta,const arma::mat&Inspec,const arma::mat&mimic,const int&Max_iter,
                          const double&Tol,const double&hn){
    int betadim=X.n_cols;
    arma::mat e10=arma::zeros(betadim,1);
    e10(0,0)=1;
    arma::mat beta10=beta0+hn*e10;
    double lg10=Profile_Main_Dep2(beta10,lambda0,X,Delta,Inspec,mimic,Max_iter,Tol);
    arma::mat beta1010=beta0+hn*e10+hn*e10;
    double lg1010=Profile_Main_Dep2(beta1010,lambda0,X,Delta,Inspec,mimic,Max_iter,Tol);
    arma::mat InfoMat=arma::zeros(betadim,betadim);
    InfoMat(0,0)=lgvalue-2*lg10+lg1010;
    InfoMat=-InfoMat/std::pow(hn,2);
    arma::mat CovMat=InfoMat.i();
    // std::cout<<lg10;
    
    arma::mat se=sqrt(CovMat.diag());
    return se;
}



