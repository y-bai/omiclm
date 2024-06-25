#!/bin/bash
#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 source_file target_file num_threads custom_header"
    exit 1
fi

source_file=$1
target_file=$2
num_threads=$3
custom_header=$4

if [ ! -f "$source_file" ]; then
    echo "Source file not found!"
    exit 1
fi

# write the first specific line into target file
echo "$custom_header" > "$target_file"

# define func to process one line
process_line() {
    line="$1"
    cleaned_line=""
    IFS=$'\t' read -r -a fields <<< "$line"
    for field in "${fields[@]}"; do
        cleaned_field=$(echo "$field" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//;s/\\//g')
        if [ -z "$cleaned_line" ]; then
            cleaned_line="$cleaned_field"
        else
            cleaned_line+=",$cleaned_field"
        fi
    done
    echo "$cleaned_line"
}

export -f process_line

# create named pipeline using mkfifo
fifo_name=$(mktemp -u)
mkfifo "$fifo_name"

# parallelly process each line using xargs, ignore the first line
tail -n +2 "$source_file" | xargs -P "$num_threads" -I {} bash -c 'process_line "{}"' > "$fifo_name" &

# write into target from named pipeline
cat "$fifo_name" >> "$target_file"

# delelte pipeline
rm "$fifo_name"

echo "Processing complete. Cleaned data written to $target_file."

# # check number of parameters
# if [ "$#" -ne 2 ]; then
#     echo "Usage: $0 source_file target_file"
#     exit 1
# fi

# source_file=$1
# target_file=$2

# # check if source_file exist
# if [ ! -f "$source_file" ]; then
#     echo "Source file not found!"
#     exit 1
# fi

# # empty target file
# > "$target_file"

# # read line
# while IFS= read -r line; do
#     # process per field in the line
#     cleaned_line=""
#     IFS=$'\t' read -r -a fields <<< "$line"
#     for field in "${fields[@]}"; do
#         # delete Space and Escape character(ESC) before and after each filed
#         cleaned_field=$(echo "$field" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//;s/\\//g')
#         # concat each cleaned filed with comma(,)
#         if [ -z "$cleaned_line" ]; then
#             cleaned_line="$cleaned_field"
#         else
#             cleaned_line+=",$cleaned_field"
#         fi
#     done
#     # write to target file
#     echo "$cleaned_line" >> "$target_file"
# done < "$source_file"
# echo "Processing complete. Cleaned data written to $target_file."

# nohup bash clean_data.sh /home/share/huadjyin/home/weiyilin/project/DNALLM/processed_data/DNA_training_input.tsv DNA_training_input.csv 50 > DNA_training_input.log & 