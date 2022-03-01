\name{Add_ci_boot}
\alias{Add_ci_boot}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Calculating standard errors and confidence intervals for the estimates via the bootstrap method.
}
\description{
Add_ci_boot is used to calculate standard errors and confidence intervals for the parameters via the bootstrap method. It is applicable for not only regression parameters but also the predicted survival probabilities. 
}
\usage{
Add_ci_boot(data,time_points,covariate_value,CItype=c("norm","basic","perc","bca"),
conf=0.95,boot.num=200,object_type=c("reg"),Max_iter=1000,Tol=1e-3)
}

\arguments{
  \item{data}{
A data frame. The first and second column record the left and right inspection time, respectively.
 For the right-censored time-to-event, the right inspection time is infinity, on the other hand for a left-censored time-to-event, the left inspection time is zero. The next three columns record the left, interval and right censoring indicators, respectively. The remaining columns  of the data frame are covariates, and only numerical covariates are acceptable. 
 For a nominal categorical variable, user should include the corresponding dummy variables columns in this data frame.
}
\item{time_points}{
A vector containing the time points where the survival probabilities are to be predicted.
}

\item{covariate_value}{A scalar or vector containing the covariates for which survival probabilities are to be predicted. 
}
   \item{CItype}{
   A vector of character strings representing the type of intervals required.
The value should be any subset of c("norm","basic","perc","bca"). The default value is c("norm","basic","perc","bca").
   }
   \item{conf}{
   A scalar  containing the confidence level of the required interval. The default value is 0.95.
   }
   \item{boot.num}{
   The number of bootstrap replicates. The default value is 200.
   }
   \item{object_type}{A vector of character strings. object_type="reg" implies bootstrap 
   confidence intervals to be calculated for the regression parameters. object_type="surv" implies
   bootstrap confidence intervals to be calculated for the survival probabilities.  object_type=c("reg","surv") implies bootstrap 
   confidence intervals to be calculated for the regression parameters and survival probabilities. 
   Once "surv" is included in object_type, the time_points and covariate_value must be specified.  The default value of object_type is "reg". 
   }
    \item{Max_iter}{
The maximum number of iterations. The default value is 1000.  
}
   \item{Tol}{
    This denotes the sum of the absolute relative differences of the estimates at two successive iterations. It is used to check the convergence of the parameter estimates. The default value is 1e-3.
   }
   

}
\value{
Function Add_ci_boot returns a list containing the following components:


\item{beta_boot_se}{The first column contains the estimate of the regression parameters. The second column contains the bootstrap standard error of the estimators.}
\item{CI_beta}{
A list containing the bootstrap confidence intervals. Each type of the confidence interval is aggregated within the same dataframe. For each dataframe, the first column is the index of covariates, the sort is the same as that in the original dataset. The second column indicates the type of confidence interval. The last two columns are the lower and upper bound of the confidence interval, respectively.
}

\item{surv_boot_se}{The first column contains predicted survival probabilities for the given time points and covariates. The second column contains the bootstrap standard error of these predicated survival probabilities.}
\item{CI_surv}{
A list containing the bootstrap confidence interval. Each type of the confidence interval is aggregated within the same dataframe. For each dataframe, the first column is the index of covariates, the sort is the same as that in the original dataset. The second column is the type of confidence interval. The last two columns are the lower and upper bound of the confidence interval, respectively.
}

}


\examples{
# specify the time points and covariates
Timepoints=c(0.5,0.6)
Covariates=c(0,1,0)


# calculate the bootstrap standard error and 
# confidence interval for both the regression 
# parameter and the predicted survival probabilities.
# Four types of confidence intervals will be
# returned and the confidence level is 0.95.
Bootresult=Add_ci_boot(mydata,time_points=Timepoints,
                       covariate_value = Covariates,
                       object_type = c("reg","surv"))

}

