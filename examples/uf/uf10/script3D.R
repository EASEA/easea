library("scatterplot3d") 
png(file = "uf10_pf.png")
makePNG <- function(pf_oprimal, problem)
{

  dir ="./";
  file1<-paste(dir,"objectives", sep="") 
#  file2<-paste(dir, pf_optimal, sep="")
  
  uf10_objectives = read.table(file1, header = FALSE, sep = " ")
#  uf10_pf_optimal = read.table(file2,header = FALSE, sep = " ")
   
  obj1 <- uf10_objectives$V1
  obj2 <- uf10_objectives$V2
  obj3 <- uf10_objectives$V3
#  obj1_opt <- uf10_pf_optimal$V1
#  obj2_opt <- uf10_pf_optimal$V2
#  obj3_opt <- uf10_pf_optimal$V3
  
  colors <- c("#999999", "#E69F00", "#56B4E9")
  colors <- colors[1]
 
  s3d<-scatterplot3d(obj1, obj2, obj3, color=colors,  grid = TRUE,
                col.grid = "grey",  pch=19, main="uf10",
                xlab="f1",
                ylab="f2",
                zlab="f3", 
            #   xlim=c(0,0.7),
            #   ylim=c(0,0.6),
            #   zlim=c(0,0.7),
                angle = 140)

}
pf_optimal <- "../pf/uf10/3D/UF10.3D.pf" 
makePNG(pf_optimal, "uf10")
dev.off()