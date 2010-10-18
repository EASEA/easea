

for(( i=0 ; i<20 ; i++ ))
do
    suf=`date +%s`
    echo $suf
    
    ./card --seed=$suf  --generateCSVFile=1 --nbGen=20
    mv card.csv out/card-$suf.csv
done