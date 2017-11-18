#! /bin/env Rscript

library(ggplot2)

dat1<-read.csv('results/accuracy_1_iter.csv')
dat2<-read.csv('results/accuracy_2_iter.csv')
dat3<-read.csv('results/accuracy_3_iter.csv')
data_dim<-dim(dat1)
dat1<-data.frame(dat1, routing_iter=factor(rep(1, data_dim[1])))
dat2<-data.frame(dat2, routing_iter=factor(rep(2, data_dim[1])))
dat3<-data.frame(dat3, routing_iter=factor(rep(3, data_dim[1])))
data<-rbind(dat1,dat2,dat3)
p<-ggplot(data,aes(x=step, y=test_acc, color=routing_iter))
p<-p + geom_line() + ylim(c(0.975, .997)) + xlim(c(0, 49000))
p<-p + labs(title='Test accuracy of different routing iterations') + theme(plot.title=element_text(hjust=0.5), legend.position=c(0.92, 0.6))
ggsave(p, filename='results/routing_trials.png', width=5, height=5)
