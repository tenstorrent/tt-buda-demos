#!/bin/bash

TIMEOUT_DURATION=3600
PYTHON_ENV_PATH="/home/anirudhsr/Releases/16_April_2024/python_env/bin/activate"
OUTPUT_DIR="./OUTPUTS"
RETRY_LIMIT=2
TEST_FILES_PATH="test_files.txt"

echo "Running script from directory: $(pwd)"

# Check if test files list exists
if [ ! -f "$TEST_FILES_PATH" ]; then
    echo "Error: Test files list $TEST_FILES_PATH does not exist."
    exit 1
fi

# Read test files into an array, removing quotes
mapfile -t TEST_FILES < <(sed 's/"//g' $TEST_FILES_PATH)

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
