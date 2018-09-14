# experiment 1 - general comparison of parametric and non-parametric activations
e=300
m=wrn
for t in '40 1' '40 2' '40 4' '16 4' '16 8' '16 10' '22 8' '22 10'; do
    d=$(echo $t | awk '{print $1}')
    k=$(echo $t | awk '{print $2}')
    for a in relu prelu aplu elu pelu blu blu-alpha blu-alpha-beta blu-const; do
        tsp bash -c "python main.py -a $a -l 1e-2 1e-6 -e $e -m $m -d $d -k $k | tee ../results/$m-$d-"$k"_"$e"_$a.out"
    done
done

# experiment 2 - residual connections removed
# for a more complete test, try all topologies from experiment 1
e=300
m=wnn
for t in '40 1' '40 2'; do
    d=$(echo $t | awk '{print $1}')
    k=$(echo $t | awk '{print $2}')
    for a in relu prelu aplu elu pelu blu blu-alpha blu-alpha-beta blu-const; do
        tsp bash -c "python main.py -a $a -l 1e-2 1e-6 -e $e -m $m -d $d -k $k | tee ../results/$m-$d-"$k"_"$e"_$a.out"
    done
done
