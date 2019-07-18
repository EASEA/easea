png(file = "zdt6_pf.png")


makePNG <- function(pf_optimal, problem)
{
	dir ="./";
	file1<-paste(dir,"objectives", sep="")
	file2<-paste(dir, pf_optimal, sep="")
	zdt6_objectives = read.table(file1, header = FALSE, sep = " ")
	zdt6_pf_optimal = read.table(file2,header = FALSE, sep = " ")

	obj1 <- zdt6_objectives$V1
	obj2 <- zdt6_objectives$V2
	obj1_opt <- zdt6_pf_optimal$V1
	obj2_opt <- zdt6_pf_optimal$V2


	 plot_colors = c("blue", "red")
  
	plot(obj1, obj2, col = "blue", xlab = "f1", ylab = "f2", xlim=c(0,1), ylim=c(0,2),  ann=T)
	lines(obj1_opt, obj2_opt, col = "red", type="l")#, lty=2, lwd=2)
	title(main="zdt6"); #titulo)
}
pf_optimal <- "../pf/zdt6/paretoZDT6M.dat"
makePNG(pf_optimal, "zdt6")


dev.off()