\name{Add_case2_inte}
\alias{Add_case2_inte}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Estimating parameters of the additive risks model with time-independent covariates and case-II interval censored data. 
}
\description{
Add_case2_inte is used to estimate the regression parameters and the cumulative hazard function for the additive risks model with interval-censored data and  time-independent covariates. 
}
\usage{
Add_case2_inte(data,hn.m,Max_iter=1000,Tol=1e-3)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{data}{
 A data frame. The first and second column record the left and right inspection time, respectively.
 For the right-censored time-to-event, the right inspection time is infinity, on the other hand for a left-censored time-to-event, the left inspection time is zero. The next three columns record the left, interval and right censoring indicators, respectively. The remaining columns  of the data frame are covariates, and only numerical covariates are acceptable. 
 For a nominal categorical variable, user should include the corresponding dummy variables columns in this data frame.
}
\item{hn.m}{
 This is the multiplier of the oscillation parameter hn  used as hn=hn.m/sqrt(n).
}
 \item{Max_iter}{
 The maximum number of iterations. The default value is 1000. 
}
   \item{Tol}{
   This denotes the sum of the absolute relative differences of the estimates at two successive iterations. It is used to check the convergence of the parameter estimates. The default value is 1e-3.
   }

}
\value{
Function Add_case2_inte returns a list containing the following components:


\item{beta}{The first column contains the estimate of the regression parameters. The second column contains the estimated standard error of these estimators. This standard errors computation makes use of the oscillation parameter hn.}
\item{lambda}{The estimate of the baseline hazard function at the unique inspection time points.
}
\item{log.likelihood}{
 The log-likelihood value.
}
\item{tk}{
 The set of the unique inspection time points.
}

}
\references{
Wang, T., Bandyopadhyay, D., and Sinha, S. (2022). \emph{Efficient estimation of the additive risks model for interval-censored data.} To appear as a book chapter.

}

\examples{

set.seed(10)
n=100

x1=rbinom(n, 1, 0.5) # the first covariate
x2=rbinom(n, 1, 0.4) # the second covariate
x3=rbinom(n, 1, 0.3) # the third covariate

#caplambda=0.2*t+ t*(0.5*x1+1*x2+0.6*x3), the true value of the regression parameters
# are 0.5, 1 and 0.6.
r=runif(n, 0, 1)
time_to_event=-log(r)/(0.2+ 0.5*x1+1*x2+0.6*x3)
# Generation of inspection time window (L, R)
myl= runif(n,0.1,1.5)
myr=runif(n, myl+1.5, 4)
#### Censoring indicator

delta_ell=as.numeric(time_to_event<myl)
delta_r=as.numeric(time_to_event>myr)
delta_i=1-delta_ell-delta_r

myr[delta_ell==1]=myl[delta_ell==1]
myl[delta_ell==1]=0
myl[delta_r==1]=myr[delta_r==1]
myr[delta_r==1]=Inf

mydata=data.frame(myl, myr, delta_ell, delta_i, delta_r, x1,x2,x3)
mydata=as.matrix(mydata)
testresult=Add_case2_inte(mydata,hn.m=1.5)



}
