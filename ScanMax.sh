#!/bin/bash
for max in {36..40}
do
    echo $max
    python Run.py $max
done

# for max in {45..55}
# do
#     echo $max
#     for c3 in $(seq -f "%.2f" 20 0.2 21.2)
#     do
#         echo $c3
#         python Run.py $max $c3
#     done
# done

# for c3 in $(seq -f "%.4f" 19.8 0.02 20.2)
# do
#     echo $c3
#     python Run.py $c3
# done

# for LAST_C3 in $(seq -f "%.2f" 20.5 0.2 21)
# do
#     echo $LAST_C3
#     for average_weight in $(seq -f "%.4f" 0.025 0.0002 0.026)
#     do
#         echo $average_weight
#         python Run.py $LAST_C3 $average_weight
#     done
# done