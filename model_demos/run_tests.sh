#!/bin/bash

TIMEOUT_DURATION=1000
PYTHON_ENV_PATH="/home/anirudhsr/Releases/16_April_2024/python_env/bin/activate"
OUTPUT_DIR="./OUTPUTS"
RETRY_LIMIT=1
MAX_TESTS=15  # Limit to run on a number

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

tmux has-session -t pytest_session 2>/dev/null
if [ $? == 0 ]; then
    echo "Session 'pytest_session' exists, killing..."
    tmux kill-session -t pytest_session
fi

tmux new-session -d -s pytest_session

TEST_FILES=$(find . -name 'test_pytorch_*.py' | head -n $MAX_TESTS)  

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
        tmux new-window -t pytest_session -n "$file_name" \
        "source $PYTHON_ENV_PATH && echo 'Virtual environment is now active at \$VIRTUAL_ENV'; echo Running tests in $test_file; timeout $TIMEOUT_DURATION python3 -m pytest $test_file -svv | tee $output_file"

        
        tmux wait-for -S ${file_name}_done

        if [ $? -eq 124 ]; then 
            echo "Test $test_file timed out or hanged, attempting to reset board and retry..."
            retries=$((retries+1))
        else
            break 
        fi
    done

    if [ $retries -gt $RETRY_LIMIT ]; then
        echo "Test $test_file failed after $RETRY_LIMIT retries."
    fi

    test_count=$((test_count+1))  
    if [ $test_count -ge $MAX_TESTS ]; then
        break  
    fi
done

# Attach to the tmux session
tmux attach-session -t pytest_session
