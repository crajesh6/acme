#/bin/bash

#
# job_path="/home/chandana/projects/acme/scripts/kld_atac_jobs.txt"
# touch $job_path
# # clear the files
# echo -n "" > $job_path
#
# long_arg="--cell_line=PC-3\
#  --model_name=blah_3\
#  --model_path=/shared/share_zenodo/trained_models/new_models/CNN/1/all/Exp/run-20220321_171001-53za5qem\
#  --quantitative_data_dir=/shared/share_zenodo/datasets/quantitative_data/cell_line_testsets/cell_line_13/complete/peak_centered/i_2048_w_1/\
#  --attr_map_path=/home/amber/saliency_repo/new_models_CNN_1_all_Exp_PC-3.pickle\
#  --base_dir=/home/chandana/projects/acme/results/atac/baseline/test\
#  --model_type=binary\
#  --evaluate_model=True"
#
#
# echo "python kld_atac.py $long_arg" >> $job_path
# data_dict = {
#         'A549': '/shared/share_zenodo/datasets/quantitative_data/cell_line_testsets/cell_line_8/complete/peak_centered/i_2048_w_1',
#         'HCT116': '/shared/share_zenodo/datasets/quantitative_data/cell_line_testsets/cell_line_9/complete/peak_centered/i_2048_w_1',
#         'GM12878': '/shared/share_zenodo/datasets/quantitative_data/cell_line_testsets/cell_line_7/complete/peak_centered/i_2048_w_1',
#         'K562': '/shared/share_zenodo/datasets/quantitative_data/cell_line_testsets/cell_line_5/complete/peak_centered/i_2048_w_1',
#         'PC-3': '/shared/share_zenodo/datasets/quantitative_data/cell_line_testsets/cell_line_13/complete/peak_centered/i_2048_w_1',
#         'HepG2': '/shared/share_zenodo/datasets/quantitative_data/cell_line_testsets/cell_line_2/complete/peak_centered/i_2048_w_1'
#  }

declare -A array
COUNT=0
array["A549"]="/shared/share_zenodo/datasets/quantitative_data/cell_line_testsets/cell_line_8/complete/peak_centered/i_2048_w_1"
array["HCT116"]="/shared/share_zenodo/datasets/quantitative_data/cell_line_testsets/cell_line_9/complete/peak_centered/i_2048_w_1"
array["GM12878"]="/shared/share_zenodo/datasets/quantitative_data/cell_line_testsets/cell_line_7/complete/peak_centered/i_2048_w_1"
array["K562"]="/shared/share_zenodo/datasets/quantitative_data/cell_line_testsets/cell_line_5/complete/peak_centered/i_2048_w_1"
array["PC-3"]="/shared/share_zenodo/datasets/quantitative_data/cell_line_testsets/cell_line_13/complete/peak_centered/i_2048_w_1"
array["HepG2"]="/shared/share_zenodo/datasets/quantitative_data/cell_line_testsets/cell_line_2/complete/peak_centered/i_2048_w_1"



binary_model=/shared/share_zenodo/trained_models/binary/*/*
bpnet=/shared/share_zenodo/trained_models/bpnet/augmentation_48/run-20211006_190817-456uzbu4
basenji_128=/shared/share_zenodo/trained_models/basenji_v2/binloss_basenji_v2/run-20220406_162758-bpl8g29s
new_model=/shared/share_zenodo/trained_models/new_models/*/*/*/*/*
quantitative_model=("${new_model[@]}" "${basenji_128[@]}" "${bpnet[@]}")
model_compile=("${binary_model[@]}" "${quantitative_model[@]}")

for entry in $model_compile
do
  # echo "$entry"
  for i in "${!array[@]}"
      do
        # echo "key  : $i"
        # echo "value: ${array[$i]}"
        COUNT=$((COUNT+1))
      done
done
echo $COUNT
