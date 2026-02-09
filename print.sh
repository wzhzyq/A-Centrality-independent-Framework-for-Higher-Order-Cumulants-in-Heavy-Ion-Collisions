#!/bin/bash

# # 方法1：使用seq
# for i in $(seq 1 10); do
#     echo $i
# done

# # 或者方法2：使用bash的扩展语法
# for i in {1..10}; do
#     echo $i
# done

# # 或者方法3：如果一定要用awk
# awk 'BEGIN{for(i=1;i<=10;i++) print i}'

# 检查文件是否存在
if [ ! -f "CBWC_Results_FromRef3/all_results.txt" ]; then
    echo "错误：文件 CBWC_Results_FromRef3/all_results.txt 不存在"
    exit 1
fi

if [ ! -f "CBWC_Results_FromNpart/all_results.txt" ]; then
    echo "错误：文件 CBWC_Results_FromNpart/all_results.txt 不存在"
    # exit 1
fi

OUTPUT_FILE="output_results.py"

# 使用tee命令同时输出到屏幕和文件，或者使用重定向只输出到文件
{
    echo "import numpy as np"
    # 获取列数
    num_cols=$(head -n 1 CBWC_Results_FromRef3/all_results.txt | wc -w)
    
    # 处理每一列
    for i in $(seq 1 $num_cols); do
        col_name=$(head -n 1 CBWC_Results_FromRef3/all_results.txt | awk -v col="$i" '{print $col}')
        # 获取该列的数据
        printf "%s=np.array([" "$col_name"
        awk -v col="$i" 'NR>1{printf("%s%s", NR==2?"":", ", $col)}' CBWC_Results_FromRef3/all_results.txt
        echo "])"
    done
    
    # # 获取列数
    # num_cols=$(head -n 1 CBWC_Results_FromNpart/all_results.txt | wc -w)
    
    # # 处理每一列
    # for i in $(seq 1 $num_cols); do
    #     # 获取列名
    #     col_name=$(head -n 1 CBWC_Results_FromNpart/all_results.txt | awk -v col="$i" '{print $col}')
    #     # 获取该列的数据
    #     printf "%s=np.array([" "$col_name"
    #     awk -v col="$i" 'NR>1{printf("%s%s", NR==2?"":", ", $col)}' CBWC_Results_FromNpart/all_results.txt
    #     echo "])"
    # done
} | tee "$OUTPUT_FILE"

echo -e "\n输出已保存到文件: $OUTPUT_FILE"


