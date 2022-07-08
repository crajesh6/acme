#/bin/bash

# this is a bash script that will make a text file for running
# kld computations on the atac-seq models
job_path="/home/chandana/projects/acme/scripts/kld_atac_jobs.txt"
touch $job_path
# clear the files
echo -n "" > $job_path

# declare the cell types
declare -A array
COUNT=0
array["A549"]="/shared/share_zenodo/datasets/quantitative_data/cell_line_testsets/cell_line_8/complete/peak_centered/i_2048_w_1"
array["HCT116"]="/shared/share_zenodo/datasets/quantitative_data/cell_line_testsets/cell_line_9/complete/peak_centered/i_2048_w_1"
array["GM12878"]="/shared/share_zenodo/datasets/quantitative_data/cell_line_testsets/cell_line_7/complete/peak_centered/i_2048_w_1"
array["K562"]="/shared/share_zenodo/datasets/quantitative_data/cell_line_testsets/cell_line_5/complete/peak_centered/i_2048_w_1"
array["PC-3"]="/shared/share_zenodo/datasets/quantitative_data/cell_line_testsets/cell_line_13/complete/peak_centered/i_2048_w_1"
array["HepG2"]="/shared/share_zenodo/datasets/quantitative_data/cell_line_testsets/cell_line_2/complete/peak_centered/i_2048_w_1"


# declare the file paths
binary_model=/shared/share_zenodo/trained_models/binary/*/*
bpnet=/shared/share_zenodo/trained_models/bpnet/augmentation_48/run-20211006_190817-456uzbu4
basenji_128=/shared/share_zenodo/trained_models/basenji_v2/binloss_basenji_v2/run-20220406_162758-bpl8g29s
new_model=/shared/share_zenodo/trained_models/new_models/*/*/*/*/*
quantitative_model=("${new_model[@]}" "${basenji_128[@]}" "${bpnet[@]}")
model_compile=("${binary_model[@]}" "${quantitative_model[@]}")

# loop through the binary models
for model_path in $binary_model
do
  for cell_type in "${!array[@]}"
      do

        long_arg="--cell_line=${cell_type}\
 --model_path=${model_path}\
 --attr_map_path=/home/amber/saliency_repo\
 --base_dir=/home/chandana/projects/acme/results/atac/baseline/binary\
 --model_type=binary\
 --evaluate_model=False"
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
 --quantitative_data_dir=$array[$i]\
 --attr_map_path=/home/amber/saliency_repo\
 --base_dir=/home/chandana/projects/acme/results/atac/baseline/quantitative\
 --model_type=quantitative\
 --evaluate_model=False"
         echo "python kld_atac.py $long_arg" >> $job_path
      done
done
