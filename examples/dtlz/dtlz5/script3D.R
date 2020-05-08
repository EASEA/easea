library("scatterplot3d") 
png(file = "dtlz5_pf.png")
makePNG <- function(pf_oprimal, problem)
{

  dir ="./";
  file1<-paste(dir,"objectives", sep="") 
  file2<-paste(dir, pf_optimal, sep="")
  
  dtlz5_objectives = read.table(file1, header = FALSE, sep = " ")
  dtlz5_pf_optimal = read.table(file2,header = FALSE, sep = " ")
   
  obj1 <- dtlz5_objectives$V1
  obj2 <- dtlz5_objectives$V2
  obj3 <- dtlz5_objectives$V3
#  obj1 <- dtlz5_pf_optimal$V1
#  obj2 <- dtlz5_pf_optimal$V2
#  obj3 <- dtlz5_pf_optimal$V3
  
  colors <- c("#999999", "#E69F00", "#56B4E9")
  colors <- colors[1]
 
  s3d<-scatterplot3d(obj1, obj2, obj3, color=colors,  grid = TRUE,
                col.grid = "grey",  pch=19, main="",
                xlab="f1",
                ylab="f2",
                zlab="f3", 
               xlim=c(0,0.8),
               ylim=c(0,0.8),
               zlim=c(0,1),
                angle = 140)

}
pf_optimal <- "../pf/dtlz5/3D/DTLZ5.3D.pf" 
makePNG(pf_optimal, "dtlz5")
dev.off()