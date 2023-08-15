# to test the speed of inference with or w/o flash attention
# loop through:
# -p in "short" "middle" "base" "long4" "long3" "long2" "long1"
# has -f or not
export HF_HOME="/data/cache/huggingface"
MODEL="65b"
OUT_DIR="test_speed"
P_LIST="short middle base long4 long3 long2 long1"

# no flash attention
for P in $P_LIST; do
    # echo "python test.py -p $P -m $MODEL > $OUT_DIR/$P-no-flash.txt"
    echo -e "\n\nStart testing $P"
    python test.py -p $P -m $MODEL > $OUT_DIR/$P-no-flash.txt
done

# with flash attention
for P in $P_LIST; do
    # echo "python test.py -p $P -m $MODEL -f > $OUT_DIR/$P-flash.txt"
    echo -e "\n\nStart testing $P"
    python test.py -p $P -m $MODEL -f > $OUT_DIR/$P-flash.txt
done
