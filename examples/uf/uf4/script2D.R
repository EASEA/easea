png(file = "uf4_pf.png")


makePNG <- function(pf_optimal, problem)
{
	dir ="./";
	file1<-paste(dir,"objectives", sep="")
	file2<-paste(dir, pf_optimal, sep="")
	uf4_objectives = read.table(file1, header = FALSE, sep = " ")
	uf4_pf_optimal = read.table(file2,header = FALSE, sep = " ")

	obj1 <- uf4_objectives$V1
	obj2 <- uf4_objectives$V2
	obj1_opt <- uf4_pf_optimal$V1
	obj2_opt <- uf4_pf_optimal$V2


	 plot_colors = c("blue", "red")
 
	plot(obj1, obj2, col = "blue", xlab = "f1", ylab = "f2")#, xlim=c(0,2), ylim=c(0,4),  ann=T)
	lines(obj1_opt, obj2_opt, col = "red", type="l")#, lty=2, lwd=2)
	title(main="uf4"); #titulo)
}
pf_optimal <- "../pf/uf4/2D/UF4.2D.pf"
makePNG(pf_optimal, "UF")


dev.off()