#!/bin/bash

TIMEOUT_DURATION=3600
PYTHON_ENV_PATH="/home/anirudhsr/Releases/16_April_2024/python_env/bin/activate"
OUTPUT_DIR="./OUTPUTS"
RETRY_LIMIT=2
echo "Running script from directory: $(pwd)"
# Explicitly define the list of test files
declare -a TEST_FILES=(
    "./tests/test_pytorch_albert.py"
    "./tests/test_pytorch_autoencoder.py"
    "./tests/test_pytorch_beit.py"
    "./tests/test_pytorch_bert.py"
    "./tests/test_pytorch_clip.py"
    "./tests/test_pytorch_codegen.py"
    "./tests/test_pytorch_deit.py"
    "./tests/test_pytorch_densenet.py"
    "./tests/test_pytorch_distilbert.py"
    "./tests/test_pytorch_dpr.py"
    "./tests/test_pytorch_falcon.py"
    "./tests/test_pytorch_flant5.py"
    "./tests/test_pytorch_fuyu8b.py"
    "./tests/test_pytorch_ghostnet.py"
    "./tests/test_pytorch_googlenet.py"
    "./tests/test_pytorch_gpt2.py"
    "./tests/test_pytorch_gptneo.py"
    "./tests/test_pytorch_hardnet.py"
    "./tests/test_pytorch_hrnet.py"
    "./tests/test_pytorch_inceptionv4.py"
    "./tests/test_pytorch_mlpmixer.py"
    "./tests/test_pytorch_mobilenetv1.py"
    "./tests/test_pytorch_mobilenetv2.py"
    "./tests/test_pytorch_mobilenetv3.py"
    "./tests/test_pytorch_openpose.py"
    "./tests/test_pytorch_opt.py"
    "./tests/test_pytorch_perceiverio.py"
    "./tests/test_pytorch_resnet.py"
    "./tests/test_pytorch_resnext.py"
    "./tests/test_pytorch_roberta.py"
    "./tests/test_pytorch_squeezebert.py"
    "./tests/test_pytorch_stable_diffusion.py"
    "./tests/test_pytorch_t5.py"
    "./tests/test_pytorch_unet.py"
    "./tests/test_pytorch_vgg.py"
    "./tests/test_pytorch_vilt.py"
    "./tests/test_pytorch_vit.py"
    "./tests/test_pytorch_vovnet.py"
    "./tests/test_pytorch_whisper.py"
    "./tests/test_pytorch_wideresnet.py"
    "./tests/test_pytorch_xception.py"
    "./tests/test_pytorch_xglm.py"
    "./tests/test_pytorch_yolov3.py"
    "./tests/test_pytorch_yolov5.py"
    "./tests/test_onnx_resnet.py"
    "./tests/test_onnx_retinanet.py"
    "./tests/test_tflite_efficientnet_lite.py"
    "./tests/test_tflite_landmark.py"
    "./tests/test_tflite_mobilenet_ssd.py"
)
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Created output directory at $OUTPUT_DIR"
fi

for test_file in "${TEST_FILES[@]}"; do
    if [ ! -f "$test_file" ]; then
        echo "Warning: Test file $test_file not found."
        continue
    fi

    file_name=$(basename "$test_file" .py)
    output_file="$OUTPUT_DIR/${file_name}.txt"
    retries=0

    while [ $retries -le $RETRY_LIMIT ]; do
        deactivate 2>/dev/null
        source $PYTHON_ENV_PATH
        echo "Running tests in $test_file (Attempt $((retries + 1)))"
        timeout $TIMEOUT_DURATION python3 -m pytest $test_file -svv | tee $output_file
        test_exit_status=$?

        if [ $test_exit_status -eq 124 ]; then 
            echo "Test $test_file timed out or hanged, retrying..."
            retries=$((retries+1))
        elif [ $test_exit_status -ne 0 ]; then
            echo "Test failed with status $test_exit_status, retrying..."
            deactivate
            retries=$((retries+1))
        else
            echo "Test $test_file completed successfully."
            break
        fi
    done

    if [ $retries -gt $RETRY_LIMIT ]; then
        echo "Test $test_file failed after $RETRY_LIMIT retries."
    fi
done

echo "All tests have been run :)"
