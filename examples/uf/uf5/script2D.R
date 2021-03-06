png(file = "uf5_pf.png")


makePNG <- function(pf_optimal, problem)
{
	dir ="./";
	file1<-paste(dir,"objectives", sep="")
	file2<-paste(dir, pf_optimal, sep="")
	uf5_objectives = read.table(file1, header = FALSE, sep = " ")
	uf5_pf_optimal = read.table(file2,header = FALSE, sep = " ")

	obj1 <- uf5_objectives$V1
	obj2 <- uf5_objectives$V2
	obj1_opt <- uf5_pf_optimal$V1
	obj2_opt <- uf5_pf_optimal$V2


	 plot_colors = c("blue", "red")
 
	plot(obj1, obj2, col = "blue", xlab = "f1", ylab = "f2")#, xlim=c(0,2), ylim=c(0,4),  ann=T)
	lines(obj1_opt, obj2_opt, col = "red", type="l")#, lty=2, lwd=2)
	title(main="uf5"); #titulo)
}
pf_optimal <- "../pf/uf5/2D/UF5.2D.pf"
makePNG(pf_optimal, "UF")


dev.off()