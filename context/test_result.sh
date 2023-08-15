# to test the sense and correctness of the results
# loop through:
# -p in "short" "middle" "base" "long4" "long3" "long2" "long1" "long"
# -i in None, "linear" "dynamic"
# -s: match the -p
#       for linear, -, -, -, 2, 2, 4, 4, 8
#       for dynamic, -, -, -, 2, 4, 6, 8, 16
# -q: 0, 1, 2, 3

MODEL="65b"
OUT_DIR="test_result"
P_LIST="short middle base long4 long3 long2 long1 long"
LINEAR_S_LIST="- - - 2 2 4 4 8"
DYNAMIC_S_LIST="- - - 2 4 6 8 16"

# none interpolation
for P in $P_LIST; do
    for Q in 0 1 2 3; do
        # echo "python test.py -p $P -q $Q -m $MODEL -f > $OUT_DIR/$P-$Q-none.txt"
        python test.py -p $P -q $Q -m $MODEL -f > $OUT_DIR/$P-$Q-none.txt
    done
done

# linear interpolation
for i in 3 4 5 6 7; do
    s=$(echo $LINEAR_S_LIST | cut -d " " -f $((i+1)))
    p=$(echo $P_LIST | cut -d " " -f $((i+1)))
    for Q in 0 1 2 3; do
        # echo "python test.py -p $p -q $Q -m $MODEL -i linear -s $s -f > $OUT_DIR/$p-$Q-linear-$s.txt"
        python test.py -p $p -q $Q -m $MODEL -i linear -s $s -f > $OUT_DIR/$p-$Q-linear-$s.txt
    done
done

# dynamic interpolation
for i in 3 4 5 6 7; do
    s=$(echo $DYNAMIC_S_LIST | cut -d " " -f $((i+1)))
    p=$(echo $P_LIST | cut -d " " -f $((i+1)))
    for Q in 0 1 2 3; do
        # echo "python test.py -p $p -q $Q -m $MODEL -i dynamic -s $s -f > $OUT_DIR/$p-$Q-dynamic-$s.txt"
        python test.py -p $p -q $Q -m $MODEL -i dynamic -s $s -f > $OUT_DIR/$p-$Q-dynamic-$s.txt
    done
done