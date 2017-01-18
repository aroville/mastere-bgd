install.packages("tree")
library(tree)

calif = read.table("http://www.stat.cmu.edu/~cshalizi/350/hw/06/cadata.dat",header=TRUE)
require(tree)
treefit = tree(log(MedianHouseValue) ~ Longitude+Latitude , data=calif)#Longitude+Latitude+MedianHouseAge,data=calif)
summary(treefit)
treefit
print(treefit)
plot(treefit)
text(treefit)



price.deciles = quantile(calif$MedianHouseValue,0:10/10)
cut.prices = cut(calif$MedianHouseValue,price.deciles,include.lowest=TRUE)
plot(calif$Longitude,calif$Latitude,col=grey(10:2/11)[cut.prices],pch=20,xlab="Longitude",ylab="Latitude")
partition.tree(treefit, label=c("Longitude","Latitude"),ordvars=c("Longitude","Latitude"),add=TRUE)
head(calif)

pruned=prune.tree(treefit,best=5)

prunedSeq=prune.tree(treefit)
plot(prunedSeq)
prunedSeq$dev

CVtree=cv.tree(treefit)
CVtree
plot(CVtree)

plot(calif$Longitude,calif$Latitude,col=grey(10:2/11)[cut.prices],pch=20,xlab="Longitude",ylab="Latitude")
partition.tree(pruned,ordvars=c("Longitude","Latitude"),add=TRUE,cex=0.3)
par(col="red")
partition.tree(treefit,ordvars=c("Longitude","Latitude"),add=TRUE,cex=0.3,col="red",lwd=0.5)
