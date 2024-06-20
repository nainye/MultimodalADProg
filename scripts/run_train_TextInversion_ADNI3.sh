export MODEL_DIR="stabilityai/stable-diffusion-2-1-base"
export OUTPUT_DIR="/inye/results/TextInversion_ADNI3_3"
export TRAIN_JSON_FILE="/inye/dataset/ADNI3_train_metadata2.jsonl"
export TEST_JSON_FILE="/inye/dataset/ADNI3_test_metadata2.jsonl"
export DATA_DIR="/inye/dataset/T1_2D_slice_512/"

cd /inye/source

CUDA_VISIBLE_DEVICES="0" accelerate launch train.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --data_json_file=$TRAIN_JSON_FILE \
 --test_data_json_file=$TEST_JSON_FILE \
 --data_dir=$DATA_DIR \
 --resolution=512 \
 --num_train_epochs=1000 \
 --learning_rate=1e-05 \
 --scale_lr \
 --lr_scheduler="constant" \
 --train_batch_size=8 \
 --dataloader_num_workers=8 \
 --save_steps=756 \
 --checkpointing_steps=756 \
 --mixed_precision="fp16" \
 --seed=0 \