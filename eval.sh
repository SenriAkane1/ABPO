export CUDA_VISIBLE_DEVICES=0,1
MODEL_NAME="ckpt"

python evaluate_model.py \
    --model_path=$MODEL_NAME \
    --dataset=di-zhang-fdu/MATH500 \
    --scale=1.5B \
    --sys=1


python evaluate_model.py \
    --model_path=$MODEL_NAME \
    --dataset=datasets/converted_aime_dataset \
    --scale=1.5B \
    --sys=1


python evaluate_model.py \
    --model_path=$MODEL_NAME \
    --dataset=zwhe99/amc23 \
    --scale=1.5B \
    --sys=1




python evaluate_model.py \
    --model_path=$MODEL_NAME \
    --dataset=math-ai/olympiadbench \
    --scale=1.5B \

