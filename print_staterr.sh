#!/bin/bash

# 定义输出文件名
output_file="formatted_Staterr_corr.py"

tag="corr_fit"

# 清空或创建输出文件
> "$output_file"

# 读取文件的第一行（标题行）以获取列名
read -r header < bootstrap_cumulant_stddevs_${tag}_tmp.txt 

# 将标题行分割为数组（使用制表符或空格分隔）
IFS=$'\t ' read -ra columns <<< "$header"

# 打印列名用于调试
echo "列名: ${columns[@]}" >&2

# 跳过标题行，处理数据行
tail -n +2 bootstrap_cumulant_stddevs_${tag}_tmp.txt   | while read -r line; do
    # 分割数据行为数组（使用制表符或空格分隔）
    IFS=$'\t ' read -ra data <<< "$line"

    # 打印数据行用于调试
    echo "数据行: ${data[@]}" >&2

    # 遍历每一列，将数据格式化为 Python 数组
    for i in "${!columns[@]}"; do
        # 提取列名（去掉前缀 "StdDev_"）
        col_name="${columns[$i]#StdDev_}"
        # 提取对应的数据（如果数据为空，跳过）
        if [[ -n "${data[$i]}" ]]; then
            echo "${col_name}_stat_err_$tag+=(${data[$i]})" >> "$output_file"
        fi
    done
done

# 为每个列名生成最终的 np.array 格式
#添加导入 numpy 库
echo "import numpy as np" >> "$output_file"
for col in "${columns[@]}"; do
    col_name="${col#StdDev_}"
    # 提取该列的所有数据
    data_values=$(grep "${col_name}_stat_err_$tag+=" "$output_file" | sed -E "s/.*${col_name}_stat_err_$tag\+=\((.*)\)/\1/" | tr '\n' ' ' | sed 's/ $//')
    # 替换为 np.array 格式（如果数据不为空）
    if [[ -n "$data_values" ]]; then
        sed -i "/${col_name}_stat_err_$tag+=/d" "$output_file"
        echo "${col_name}_stat_err_$tag = np.array([$(echo "$data_values" | sed 's/ /, /g')])" >> "$output_file"
    else
        sed -i "/${col_name}_stat_err_$tag+=/d" "$output_file"
        echo "${col_name}_stat_err_$tag = np.array([])" >> "$output_file"
    fi
done

echo "数据已格式化并保存到 $output_file"