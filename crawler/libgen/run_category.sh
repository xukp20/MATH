# go through LEN of i
LEN=33
for ((i=0;i<LEN;i++))
    # call python script
    do python run_category.py -i $i
done
