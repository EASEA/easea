png(file = "zdt2_pf.png")


makePNG <- function(pf_optimal, problem)
{
	dir ="./";
	file1<-paste(dir,"objectives", sep="")
	file2<-paste(dir, pf_optimal, sep="")
	zdt2_objectives = read.table(file1, header = FALSE, sep = " ")
	zdt2_pf_optimal = read.table(file2,header = FALSE, sep = " ")

	obj1 <- zdt2_objectives$V1
	obj2 <- zdt2_objectives$V2
	obj1_opt <- zdt2_pf_optimal$V1
	obj2_opt <- zdt2_pf_optimal$V2


	 plot_colors = c("blue", "red")
  
	plot(obj1, obj2, col = "blue", xlab = "f1", ylab = "f2", xlim=c(0,1), ylim=c(0,2),  ann=T)
	lines(obj1_opt, obj2_opt, col = "red", type="l")#, lty=2, lwd=2)
	title(main="zdt2"); #titulo)
}
pf_optimal <- "../pf/zdt2/paretoZDT2.dat"
makePNG(pf_optimal, "zdt2")


dev.off()