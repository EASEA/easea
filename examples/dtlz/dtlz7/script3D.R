library("scatterplot3d") 
png(file = "dtlz7_pf.png")
makePNG <- function(pf_oprimal, problem)
{

  dir ="./";
  file1<-paste(dir,"objectives", sep="") 
  file2<-paste(dir, pf_optimal, sep="")
  
  dtlz7_objectives = read.table(file1, header = FALSE, sep = " ")
  dtlz7_pf_optimal = read.table(file2,header = FALSE, sep = " ")
   
  obj1 <- dtlz7_objectives$V1
  obj2 <- dtlz7_objectives$V2
  obj3 <- dtlz7_objectives$V3
#  obj1 <- dtlz7_pf_optimal$V1
#  obj2 <- dtlz7_pf_optimal$V2
#  obj3 <- dtlz7_pf_optimal$V3
  
  colors <- c("#999999", "#E69F00", "#56B4E9")
  colors <- colors[1]
 
  s3d<-scatterplot3d(obj1, obj2, obj3, color=colors,  grid = TRUE,
                col.grid = "grey",  pch=19, main="",
                xlab="f1",
                ylab="f2",
                zlab="f3", 
               xlim=c(0,1),
               ylim=c(0,1),
               zlim=c(2.5,6),
                angle = 140)

}
pf_optimal <- "../pf/dtlz7/3D/DTLZ7.3D.pf" 
makePNG(pf_optimal, "dtlz7")
dev.off()