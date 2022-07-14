#/bin/bash

# this is a bash script that will make a text file for running
# kld computations on the atac-seq models
DATA_DIR="/home/shush/share_zenodo"
MODEL_DIR="/home/chandana/data/trained_models"
OUTPUT_DIR="/home/chandana/projects/acme/results/atac"

# declare radius count cutoff
RCC=0.003
GPU=0

job_path="/home/chandana/projects/acme/scripts/kld_atac_jobs_series_RCC_${RCC}.txt"
touch $job_path
# clear the files
echo -n "" > $job_path


# declare the cell types
declare -A array
COUNT=0
array["A549"]="${DATA_DIR}/datasets/quantitative_data/cell_line_testsets/cell_line_8/complete/peak_centered/i_2048_w_1"
array["HCT116"]="${DATA_DIR}/datasets/quantitative_data/cell_line_testsets/cell_line_9/complete/peak_centered/i_2048_w_1"
array["GM12878"]="${DATA_DIR}/datasets/quantitative_data/cell_line_testsets/cell_line_7/complete/peak_centered/i_2048_w_1"
array["K562"]="${DATA_DIR}/datasets/quantitative_data/cell_line_testsets/cell_line_5/complete/peak_centered/i_2048_w_1"
array["PC-3"]="${DATA_DIR}/datasets/quantitative_data/cell_line_testsets/cell_line_13/complete/peak_centered/i_2048_w_1"
array["HepG2"]="${DATA_DIR}/datasets/quantitative_data/cell_line_testsets/cell_line_2/complete/peak_centered/i_2048_w_1"


# declare the file paths
binary_model="${MODEL_DIR}/binary/*/*"
bpnet="${MODEL_DIR}/bpnet/augmentation_48/run-20211006_190817-456uzbu4"
basenji_128="${MODEL_DIR}/basenji_v2/binloss_basenji_v2/run-20220406_162758-bpl8g29s"
new_model="${MODEL_DIR}/new_models/*/*/*/*/*"

# loop through the binary models
for model_path in $binary_model
do
  for cell_type in "${!array[@]}"
      do

        long_arg="--cell_line=${cell_type}\
 --model_path=${model_path}\
 --data_dir=${array[$cell_type]}\
 --attr_map_path=/home/amber/saliency_repo\
 --radius_count_cutoff=${RCC}\
 --base_dir=${OUTPUT_DIR}/${RCC}/binary\
 --model_type=binary\
 --evaluate_model=False\
 --plot_acme=False\
 --gpu=${GPU}"
         echo "python kld_atac.py $long_arg" >> $job_path
      done
done

# loop through the quantitative models
for model_path in $bpnet $basenji_128 $new_model
do
  for cell_type in "${!array[@]}"
      do

        long_arg="--cell_line=$cell_type\
 --model_path=$model_path\
 --data_dir=${array[$cell_type]}\
 --attr_map_path=/home/amber/saliency_repo\
 --radius_count_cutoff=${RCC}\
 --base_dir=${OUTPUT_DIR}/${RCC}/quantitative\
 --model_type=quantitative\
 --evaluate_model=False\
 --plot_acme=False\
 --gpu=${GPU}"
         echo "python kld_atac.py $long_arg" >> $job_path
      done
done
