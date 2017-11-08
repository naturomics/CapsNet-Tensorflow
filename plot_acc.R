#! /bin/env Rscript

library(ggplot2)

data<-read.csv(file='results/accuracy.csv',header=T)
png('results/accuracy.png')
p<-ggplot(data,aes(x=step,y=test_acc))
p<-p + geom_line(color='red') + ylim(c(0.945,0.998))
p<-p + labs(x='training step', y='test accuracy', title='test accuracy for mnist')
p + geom_abline(slope=0, intercept=max(data[,2]), color='blue')
dev.off()
