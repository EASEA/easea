run=50
f_line=2
l_line=$(($run+$f_line))

echo "Best" > concat.csv
k=4
for(( j=$f_line ; j< $l_line ; j++))
do
    for i in `ls ./card*.csv`
    do
	echo -n `head $i -n $j | tail -n1 | cut -d"," -f $k`, >> concat.csv
    done
    echo "" >> concat.csv
done

echo "Avg" >> concat.csv
k=5
for(( j=$f_line ; j< $l_line ; j++))
do
    for i in `ls ./card*.csv`
    do
	echo -n `head $i -n $j | tail -n1 | cut -d"," -f $k`, >> concat.csv
    done
    echo "" >> concat.csv
done

echo "StdDev" >> concat.csv
k=6
for(( j=$f_line ; j< $l_line ; j++))
do
    for i in `ls ./card*.csv`
    do
	echo -n `head $i -n $j | tail -n1 | cut -d"," -f $k`, >> concat.csv
    done
    echo "" >> concat.csv
done
