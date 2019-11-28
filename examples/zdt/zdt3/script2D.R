png(file = "zdt3_pf.png")


makePNG <- function(pf_optimal, problem)
{
	dir ="./";
	file1<-paste(dir,"objectives", sep="")
	file2<-paste(dir, pf_optimal, sep="")
	zdt3_objectives = read.table(file1, header = FALSE, sep = " ")
	zdt3_pf_optimal = read.table(file2,header = FALSE, sep = " ")

	obj1 <- zdt3_objectives$V1
	obj2 <- zdt3_objectives$V2
	obj1_opt <- zdt3_pf_optimal$V1
	obj2_opt <- zdt3_pf_optimal$V2


	 plot_colors = c("blue", "red")
  
	plot(obj1, obj2, col = "blue", xlab = "f1", ylab = "f2", xlim=c(0,1), ylim=c(0,2),  ann=T)
	lines(obj1_opt, obj2_opt, col = "red", type="l")#, lty=2, lwd=2)
	title(main="zdt3"); #titulo)
}
pf_optimal <- "../pf/zdt3/paretoZDT3.dat"
makePNG(pf_optimal, "zdt3")


dev.off()