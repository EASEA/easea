png(file = "zdt4_pf.png")


makePNG <- function(pf_optimal, problem)
{
	dir ="./";
	file1<-paste(dir,"objectives", sep="")
	file2<-paste(dir, pf_optimal, sep="")
	zdt4_objectives = read.table(file1, header = FALSE, sep = " ")
	zdt4_pf_optimal = read.table(file2,header = FALSE, sep = " ")

	obj1 <- zdt4_objectives$V1
	obj2 <- zdt4_objectives$V2
	obj1_opt <- zdt4_pf_optimal$V1
	obj2_opt <- zdt4_pf_optimal$V2


	 plot_colors = c("blue", "red")
  
	plot(obj1, obj2, col = "blue", xlab = "f1", ylab = "f2", xlim=c(0,1), ylim=c(0,2),  ann=T)
	lines(obj1_opt, obj2_opt, col = "red", type="l")#, lty=2, lwd=2)
	title(main="ZDT4"); #titulo)
}
pf_optimal <- "../pf/zdt4/paretoZDT4.dat"
makePNG(pf_optimal, "ZDT4")


dev.off()