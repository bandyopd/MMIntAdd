.boot_reg=function(data,indices,Max_iter=1000,Tol=1e-3){
  data=data[indices,]
  Delta=data[,3:5]
  X=as.matrix(data[,6:ncol(data)])
  n=nrow(X)
  Inspec=matrix(0,nrow=n,ncol=2)
  mimic=matrix(0,nrow=n,ncol=2)
  rawInspec=data[,1:2]
  minInspec=min(rawInspec[rawInspec>0])
  for(i in 1:n){
    if(Delta[i,1]==1){
      Inspec[i,1]=data[i,2]
      mimic[i,]=Inspec[i,]
      mimic[i,2]=Inspec[i,1]+1
    }
    else if(Delta[i,3]==1){
      Inspec[i,2]=data[i,1]
      mimic[i,]=Inspec[i,]
      mimic[i,1]=minInspec
    }
    else{
      Inspec[i,]=data[i,1:2]
      mimic[i,]=Inspec[i,1:2]
    }
  }
  options(warn=-1)
  result1C=Main_func_cpp(X,Delta,Inspec,mimic,Max_iter,Tol)
  options(warn=0)
  betadim=ncol(X)
  betaest=result1C[[1]]
  
  
  if(is.na(result1C[[1]][1])){
    data[,1:2]=data[,1:2]/10
    Delta=data[,3:5]
    X=as.matrix(data[,6:ncol(data)])
    n=nrow(X)
    Inspec=matrix(0,nrow=n,ncol=2)
    mimic=matrix(0,nrow=n,ncol=2)
    rawInspec=data[,1:2]
    minInspec=min(rawInspec[rawInspec>0])
    for(i in 1:n){
      if(Delta[i,1]==1){
        Inspec[i,1]=data[i,2]
        mimic[i,]=Inspec[i,]
        mimic[i,2]=Inspec[i,1]+1
      }
      else if(Delta[i,3]==1){
        Inspec[i,2]=data[i,1]
        mimic[i,]=Inspec[i,]
        mimic[i,1]=minInspec
      }
      else{
        Inspec[i,]=data[i,1:2]
        mimic[i,]=Inspec[i,1:2]
      }
    }
    
    result1C=Main_func_cpp(X,Delta,Inspec,mimic,Max_iter,Tol)
    betadim=ncol(X)
    betaest=result1C[[1]]
    
  }
  
  
  return(betaest[,1])
}
.boot_surv=function(data,indices,time_points,covariate_value,Max_iter=1000,Tol=1e-3){
  data=data[indices,]
  Delta=data[,3:5]
  X=as.matrix(data[,6:ncol(data)])
  n=nrow(X)
  Inspec=matrix(0,nrow=n,ncol=2)
  mimic=matrix(0,nrow=n,ncol=2)
  rawInspec=data[,1:2]
  minInspec=min(rawInspec[rawInspec>0])
  for(i in 1:n){
    if(Delta[i,1]==1){
      Inspec[i,1]=data[i,2]
      mimic[i,]=Inspec[i,]
      mimic[i,2]=Inspec[i,1]+1
    }
    else if(Delta[i,3]==1){
      Inspec[i,2]=data[i,1]
      mimic[i,]=Inspec[i,]
      mimic[i,1]=minInspec
    }
    else{
      Inspec[i,]=data[i,1:2]
      mimic[i,]=Inspec[i,1:2]
    }
  }
  
  result1C=Main_func_cpp(X,Delta,Inspec,mimic,Max_iter,Tol)
  betadim=ncol(X)
  betaest=result1C[[1]]
  lambdaest=exp(result1C[[2]])
  log_likelihood=as.numeric(result1C[[3]])
  betaresult=matrix(0, nrow = betadim, ncol = 2)
  betaresult[,1]=betaest
  colnames(betaresult)=c("Est","SE")
  tk=sort(unique(c(unique(data[,1]),unique(data[,2]))))
  tk=tk[-c(1,length(tk))]
  
  time_length=length(time_points)
  
  
  Survival_pred=c()
  for (i in 1:time_length) {
    Lambda=sum(lambdaest[lambdaest<=time_points[i]])
    
    Survival_pred=append(Survival_pred,-Lambda-sum(betaest*covariate_value)*time_points[i])
  }
  
  
  
  if(is.na(result1C[[1]][1])){
    data[,1:2]=data[,1:2]/10
    Delta=data[,3:5]
    X=as.matrix(data[,6:ncol(data)])
    n=nrow(X)
    Inspec=matrix(0,nrow=n,ncol=2)
    mimic=matrix(0,nrow=n,ncol=2)
    rawInspec=data[,1:2]
    minInspec=min(rawInspec[rawInspec>0])
    for(i in 1:n){
      if(Delta[i,1]==1){
        Inspec[i,1]=data[i,2]
        mimic[i,]=Inspec[i,]
        mimic[i,2]=Inspec[i,1]+1
      }
      else if(Delta[i,3]==1){
        Inspec[i,2]=data[i,1]
        mimic[i,]=Inspec[i,]
        mimic[i,1]=minInspec
      }
      else{
        Inspec[i,]=data[i,1:2]
        mimic[i,]=Inspec[i,1:2]
      }
    }
    
    result1C=Main_func_cpp(X,Delta,Inspec,mimic,Max_iter,Tol)
    betadim=ncol(X)
    betaest=result1C[[1]]
    lambdaest=exp(result1C[[2]])
    log_likelihood=as.numeric(result1C[[3]])
    betaresult=matrix(0, nrow = betadim, ncol = 2)
    betaresult[,1]=betaest
    colnames(betaresult)=c("Est","SE")
    tk=sort(unique(c(unique(data[,1]),unique(data[,2]))))
    tk=tk[-c(1,length(tk))]
    time_length=length(time_points)
    
    
    Survival_pred=c()
    for (i in 1:time_length) {
      Lambda=sum(lambdaest[lambdaest<=time_points[i]])
      
      Survival_pred=append(Survival_pred,-Lambda-sum(betaest*covariate_value)*time_points[i])
    }
    
  }
  
  return(Survival_pred)
  
}

.getCI <- function(w,x,CItype,conf) {
  b1 <- suppressWarnings(boot::boot.ci(x,type=CItype,conf=conf,index=w))
  ## extract info for all CI types
  tab <- t(sapply(b1[-(1:3)],function(x) tail(c(x),2)))
  ## combine with metadata: CI method, index
  tab <- cbind(w,rownames(tab),as.data.frame(tab))
  colnames(tab) <- c("index","method","lwr","upr")
  tab
}

.exp.matrix=function(rawresult){
  methodlist=unique(rawresult$method)
  num.method=length(methodlist)
  finalresult=list()
  for (i in methodlist) {
    finalresult[[i]]=rawresult[rawresult$method==i,]
  }
  return(finalresult)
}


Add_ci_boot=function(data,time_points,covariate_value,CItype=c("norm","basic","perc","bca"),conf=0.95,boot.num=200,object_type=c("reg"),Max_iter=1000,Tol=1e-3){
  if(length(object_type)==2){
    boot_reg_out=boot::boot(data,.boot_reg,boot.num,Max_iter=Max_iter,Tol=Tol)
    boot_surv_out=boot::boot(data,.boot_surv,boot.num,time_points=time_points,covariate_value=covariate_value,Max_iter=Max_iter,Tol=Tol)
    betadim=ncol(data)-5
    time_length=length(time_points)
    CI_reg=do.call(rbind,lapply(1:betadim, .getCI,x=boot_reg_out,CItype=CItype,conf=conf))
    CI_surv=do.call(rbind,lapply(1:time_length, .getCI,x=boot_surv_out,CItype=CItype,conf=conf))
    CI_surv[,c(3,4)]=exp(CI_surv[,c(3,4)])
    CI_surv[CI_surv[,4]>1,4]=1
    reg_sd=apply(boot_reg_out$t, 2, sd,na.rm=TRUE)
    surv_sd=apply(exp(boot_surv_out$t), 2, sd,na.rm=TRUE)
    beta_boot_se=as.data.frame(cbind(boot_reg_out$t0,reg_sd))
    surv_boot_se=as.data.frame(cbind(exp(boot_surv_out$t0),surv_sd))
    
    colnames(beta_boot_se)=colnames(surv_boot_se)=c("Est","boot_se")
    rownames(beta_boot_se)=tail(colnames(data),betadim)
    
    result=list(beta_boot_se=beta_boot_se,CI_beta=.exp.matrix(CI_reg),surv_boot_se=surv_boot_se,CI_surv=.exp.matrix(CI_surv))
  }else if(object_type=="reg"){
    
    
    options(warn=-1)
    boot_reg_out=boot::boot(data,.boot_reg,boot.num,Max_iter=Max_iter,Tol=Tol)
    
    options(warn=0)
    betadim=ncol(data)-5
    CI_reg=do.call(rbind,lapply(1:betadim, .getCI,x=boot_reg_out,CItype=CItype,conf=conf))
    reg_sd=apply(boot_reg_out$t, 2, sd,na.rm=TRUE)
    beta_boot_se=as.data.frame(cbind(boot_reg_out$t0,reg_sd))
    
    colnames(beta_boot_se)=c("Est","boot_se")
    rownames(beta_boot_se)=tail(colnames(data),betadim)
    result=list(beta_boot_se=beta_boot_se,CI_beta=.exp.matrix(CI_reg))
  }else if(object_type=="surv"){
    boot_surv_out=boot::boot(data,.boot_surv,boot.num,time_points=time_points,covariate_value=covariate_value,Max_iter=Max_iter,Tol=Tol)
    time_length=length(time_points)
    CI_surv=do.call(rbind,lapply(1:time_length, .getCI,x=boot_surv_out,CItype=CItype,conf=conf))
    CI_surv[,c(3,4)]=exp(CI_surv[,c(3,4)])
    CI_surv[CI_surv[,4]>1,4]=1
    
    surv_sd=apply(exp(boot_surv_out$t), 2, sd,na.rm=TRUE)
    surv_boot_se=as.data.frame(cbind(exp(boot_surv_out$t0),surv_sd))    
    colnames(surv_boot_se)=c("Est","boot_se")
    result=list(surv_boot_se=surv_boot_se,CI_surv=.exp.matrix(CI_surv))
    
  }
  return(result)
}

Add_case2_inte=function(data,hn.m,Max_iter=1000,Tol=1e-3){
  Delta=data[,3:5]
  X=as.matrix(data[,6:ncol(data)])
  n=nrow(X)
  Inspec=matrix(0,nrow=n,ncol=2)
  mimic=matrix(0,nrow=n,ncol=2)
  rawInspec=data[,1:2]
  minInspec=min(rawInspec[rawInspec>0])
  for(i in 1:n){
    if(Delta[i,1]==1){
      Inspec[i,1]=data[i,2]
      mimic[i,]=Inspec[i,]
      mimic[i,2]=Inspec[i,1]+1
    }
    else if(Delta[i,3]==1){
      Inspec[i,2]=data[i,1]
      mimic[i,]=Inspec[i,]
      mimic[i,1]=minInspec
    }
    else{
      Inspec[i,]=data[i,1:2]
      mimic[i,]=Inspec[i,1:2]
    }
  }
  
  result1C=Main_func_cpp(X,Delta,Inspec,mimic,Max_iter,Tol)
  betadim=ncol(X)
  hn=hn.m*n^(-1/2)
  # if(betadim==1){
  #   seC=SE_cal_cpp_one(matrix(result1C[[1]]),matrix(result1C[[2]]),as.numeric(result1C[[3]]),X,Delta,Inspec,mimic,1000,1e-3,hn)
  # }
  # else{
  #   seC=SE_cal_cpp(matrix(result1C[[1]]),matrix(result1C[[2]]),as.numeric(result1C[[3]]),X,Delta,Inspec,mimic,1000,1e-3,hn)
  # }
  seC=SE_cal_cpp_gen(matrix(result1C[[1]]),matrix(result1C[[2]]),as.numeric(result1C[[3]]),X,Delta,Inspec,mimic,1000,1e-3,hn)
  betaest=result1C[[1]]
  lambdaest=exp(result1C[[2]])
  log_likelihood=as.numeric(result1C[[3]])
  betaresult=matrix(0, nrow = betadim, ncol = 2)
  betaresult[,1]=betaest
  betaresult[,2]=seC
  colnames(betaresult)=c("Est","SE")
  rownames(betaresult)=colnames(data)[6:length(colnames(data))]
  tk=sort(unique(c(unique(data[,1]),unique(data[,2]))))
  tk=tk[-c(1,length(tk))]
  
  lambda_length=length(tk)
  S1=S0=c()
  for (i in 1:lambda_length) {
    Lambda=sum(lambdaest[1:i])
    S1=append(S1,exp(-Lambda-betaest*tk[i]))
    S0=append(S0,exp(-Lambda))
  }
  Est_surv=cbind(S1,S0,tk)
  finalresult=list(beta=betaresult,lambda=lambdaest,log.likelihood=log_likelihood,tk=tk,Est_surve=Est_surv)
  
  
  
  if(is.na(result1C[[1]][1])){
    data[,1:2]=data[,1:2]/10
    Delta=data[,3:5]
    X=as.matrix(data[,6:ncol(data)])
    n=nrow(X)
    Inspec=matrix(0,nrow=n,ncol=2)
    mimic=matrix(0,nrow=n,ncol=2)
    rawInspec=data[,1:2]
    minInspec=min(rawInspec[rawInspec>0])
    for(i in 1:n){
      if(Delta[i,1]==1){
        Inspec[i,1]=data[i,2]
        mimic[i,]=Inspec[i,]
        mimic[i,2]=Inspec[i,1]+1
      }
      else if(Delta[i,3]==1){
        Inspec[i,2]=data[i,1]
        mimic[i,]=Inspec[i,]
        mimic[i,1]=minInspec
      }
      else{
        Inspec[i,]=data[i,1:2]
        mimic[i,]=Inspec[i,1:2]
      }
    }
    
    result1C=Main_func_cpp(X,Delta,Inspec,mimic,Max_iter,Tol)
    betadim=ncol(X)
    hn=hn.m*n^(-1/2)
    # if(betadim==1){
    #   seC=SE_cal_cpp_one(matrix(result1C[[1]]),matrix(result1C[[2]]),as.numeric(result1C[[3]]),X,Delta,Inspec,mimic,1000,1e-3,hn)
    # }
    # else{
    #   seC=SE_cal_cpp(matrix(result1C[[1]]),matrix(result1C[[2]]),as.numeric(result1C[[3]]),X,Delta,Inspec,mimic,1000,1e-3,hn)
    # }
    seC=SE_cal_cpp_gen(matrix(result1C[[1]]),matrix(result1C[[2]]),as.numeric(result1C[[3]]),X,Delta,Inspec,mimic,1000,1e-3,hn)
    betaest=result1C[[1]]
    lambdaest=exp(result1C[[2]])
    log_likelihood=as.numeric(result1C[[3]])
    betaresult=matrix(0, nrow = betadim, ncol = 2)
    betaresult[,1]=betaest
    betaresult[,2]=seC
    colnames(betaresult)=c("Est","SE")
    rownames(betaresult)=colnames(data)[6:length(colnames(data))]
    tk=sort(unique(c(unique(data[,1]),unique(data[,2]))))
    tk=tk[-c(1,length(tk))]
    lambda_length=length(tk)
    S1=S0=c()
    for (i in 1:lambda_length) {
      Lambda=sum(lambdaest[1:i])
      S1=append(S1,exp(-Lambda-betaest*tk[i]))
      S0=append(S0,exp(-Lambda))
    }
    Est_surv=cbind(S1,S0,tk*10)
    finalresult=list(beta=betaresult,lambda=lambdaest,log.likelihood=log_likelihood,tk=tk*10,Est_surv=Est_surv)
    
  }
  
  
  
  return(finalresult)
}



