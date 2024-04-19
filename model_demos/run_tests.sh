#!/bin/bash

TIMEOUT_DURATION=3600
PYTHON_ENV_PATH="/home/anirudhsr/Releases/16_April_2024/python_env/bin/activate"
OUTPUT_DIR="./OUTPUTS"
RETRY_LIMIT=2
MAX_TESTS=15 # Limit to run on a number
ALL=true # Set to true to run all tests, otherwise limited to number set above in MAX_TESTS 

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

if [ "$ALL" = true ]; then
    TEST_FILES=$(find . -name 'test_pytorch_*.py')
else
    TEST_FILES=$(find . -name 'test_pytorch_*.py' | head -n $MAX_TESTS)
fi

if [ -z "$TEST_FILES" ]; then
    echo "No test files found."
    exit 1
fi

test_count=0

for test_file in $TEST_FILES; do
    file_name=$(basename "$test_file" .py)
    output_file="$OUTPUT_DIR/${file_name}.txt"
    retries=0

    while [ $retries -le $RETRY_LIMIT ]; do
        # Ensure virtual environment is deactivated before activating it again
        deactivate 2>/dev/null
        source $PYTHON_ENV_PATH
        echo "Virtual environment is now active at $VIRTUAL_ENV"
        echo "Running tests in $test_file"
        
        timeout $TIMEOUT_DURATION python3 -m pytest $test_file -svv | tee $output_file

        if [ $? -eq 124 ]; then 
            echo "Test $test_file timed out or hanged, attempting to reset board and retry..."
            retries=$((retries+1))
        elif [ $? -ne 0 ]; then
            echo "Test failed, clearing environment..."
            deactivate
            # Reset or clear operations here if needed
            retries=$((retries+1))
        else
            break
        fi
    done

    if [ $retries -gt $RETRY_LIMIT ]; then
        echo "Test $test_file failed after $RETRY_LIMIT retries."
    fi

    test_count=$((test_count+1))
    if [ "$ALL" = false ] && [ $test_count -ge $MAX_TESTS ]; then
        break
    fi
done
