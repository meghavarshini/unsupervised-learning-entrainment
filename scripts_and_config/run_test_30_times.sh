#!/bin/bash

# Pass test.py to variable
SCRIPT_DIR="./Asist3_data_management"
PYTHON_SCRIPT="asist3_test.py"
# Output file
OUTPUT_FILE="test_30_runs_$(date +'%Y-%m-%d_%H-%M-%S').txt"

# Change to the script directory
cd "$SCRIPT_DIR" || { echo "Failed to enter directory $SCRIPT_DIR"; exit 1; }

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --output) OUTPUT_FILE="$2"; shift ;;
        *) PYTHON_ARGS="$PYTHON_ARGS $1" ;;  # Collect Python arguments
    esac
    shift
done

echo "Directory: $PWD"
echo "Python arguments: $PYTHON_ARGS"

# Ensure the output file is empty
> "$OUTPUT_FILE"

# Run the script 30 times
for i in {1..30}; do
    # Run the Python script with the collected arguments and capture output and error
    OUTPUT=$(python "$PYTHON_SCRIPT" $PYTHON_ARGS 2>&1)
    EXIT_STATUS=$?
    
    # Check if the Python script encountered an error
    if [ $EXIT_STATUS -ne 0 ]; then
        echo "Error encountered in run $i: $OUTPUT"
        echo "Terminating script."
        exit 1
    fi
    
    # Capture only the last line of output
    LAST_LINE=$(echo "$OUTPUT" | tail -n 1)

    # Append the last line to the output file
    echo "$LAST_LINE" >> "$OUTPUT_FILE"
    echo "Run $i completed."
done

# Return to the original directory
cd - > /dev/null
echo "All runs completed. Results saved in $OUTPUT_FILE"
