library('sem')
setwd("/Volumes/16 DOS/R_nbs")
logitML<-read.csv("LogitML.csv",sep=",",header=TRUE,fileEncoding="latin1")
logitML<-logitML[1:98,]
for (i in c(1:31)){
  logitML[,i]<-as.numeric(logitML[,i])}
logitML2<-logitML[,c(29,26,24)]
SEM<-tsls(logitML2$c0_recomend~ logitML2$c0_honesto ,
          instruments=~logitML2$c0_atencao,data=logitML2)
summary(SEM)
model.dhp<-matrix(c("logitML2$c0_recomend<-logitML2$c0_honesto","gam0",NA,
                        "logitML2$c0_recomend<-logitML2$c0_atencao","gam1", NA,
                        "logitML2$c0_honesto<->logitML2$c0_honesto","gam2", 1,
                        "logitML2$c0_atencao<->logitML2$c0_atencao","gam3", 1,
                        "logitML2$c0_recomend<->logitML2$c0_recomend","gam4", 1),ncol=3, byrow=TRUE)
## covariance matrix
R.dhp <- matrix(cov(logitML2),3, 3, byrow=TRUE)
rownames(R.dhp) <- colnames(R.dhp) <-c("logitML2$c0_recomend","logitML2$c0_atencao","logitML2$c0_honesto")

SEM_2<-sem(model.dhp,R.dhp,N=98,maxiter=50)
SEM_2
standardizedCoefficients(SEM_2)
standardizedResiduals(SEM_2)
summary(SEM_2,digits=3)
