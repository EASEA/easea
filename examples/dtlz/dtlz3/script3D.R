library("scatterplot3d") 
png(file = "dtlz3_pf.png")
makePNG <- function(pf_oprimal, problem)
{

  dir ="./";
  file1<-paste(dir,"objectives", sep="") 
  file2<-paste(dir, pf_optimal, sep="")
  
  dtlz3_objectives = read.table(file1, header = FALSE, sep = " ")
  dtlz3_pf_optimal = read.table(file2,header = FALSE, sep = " ")
   
  obj1 <- dtlz3_objectives$V1
  obj2 <- dtlz3_objectives$V2
  obj3 <- dtlz3_objectives$V3
 # obj1 <- dtlz3_pf_optimal$V1
 # obj2 <- dtlz3_pf_optimal$V2
 # obj3 <- dtlz3_pf_optimal$V3
  
  colors <- c("#999999", "#E69F00", "#56B4E9")
  colors <- colors[1]
 
  s3d<-scatterplot3d(obj1, obj2, obj3, color=colors,  grid = TRUE,
                col.grid = "grey",  pch=19, main="",
                xlab="f1",
                ylab="f2",
                zlab="f3", 
               xlim=c(0,1.0),
               ylim=c(0,1.0),
               zlim=c(0,1.4),
                angle = 140)

}
pf_optimal <- "../pf/dtlz3/3D/DTLZ3.3D.pf" 
makePNG(pf_optimal, "dtlz3")
dev.off()