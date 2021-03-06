png(file = "uf7_pf.png")


makePNG <- function(pf_optimal, problem)
{
	dir ="./";
	file1<-paste(dir,"objectives", sep="")
	file2<-paste(dir, pf_optimal, sep="")
	uf7_objectives = read.table(file1, header = FALSE, sep = " ")
	uf7_pf_optimal = read.table(file2,header = FALSE, sep = " ")

	obj1 <- uf7_objectives$V1
	obj2 <- uf7_objectives$V2
	obj1_opt <- uf7_pf_optimal$V1
	obj2_opt <- uf7_pf_optimal$V2


	 plot_colors = c("blue", "red")
 
	plot(obj1, obj2, col = "blue", xlab = "f1", ylab = "f2")#, xlim=c(0,2), ylim=c(0,4),  ann=T)
	lines(obj1_opt, obj2_opt, col = "red", type="l")#, lty=2, lwd=2)
	title(main="uf7"); #titulo)
}
pf_optimal <- "../pf/uf7/2D/UF7.2D.pf"
makePNG(pf_optimal, "UF")


dev.off()