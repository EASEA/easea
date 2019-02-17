set term png
set output "memetic_weierstrass.png"
set xrange[0:1000]
set xlabel "Number of Evaluations"
set ylabel "Fitness"
plot 'memetic_weierstrass.dat' using 3:4 t 'Best Fitness' w lines, 'memetic_weierstrass.dat' using 3:5 t  'Average' w lines, 'memetic_weierstrass.dat' using 3:6 t 'StdDev' w lines
