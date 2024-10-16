#!/bin/bash

OUTPUT_DIR="${1}"
MODEL_NAME="${2}"
INPUT_FILE="${3}"

output_file_counter=0

test_input_string () {
	output_file_counter=$((output_file_counter+1))
	OUTPUT_DIR="${1}"
	MODEL_NAME="${2}"
	INPUT_STRING="${3}"
	
	output_file_name="${OUTPUT_DIR}/candidate-`printf \"%06d\" ${output_file_counter}`.txt"
	echo "${output_file_name}: [${MODEL_NAME}] ${INPUT_STRING}"
	echo "Full input string: '${INPUT_STRING}'" > "${output_file_name}" 2>&1
# sed command courtesy of https://superuser.com/a/380778
# the </dev/null is necessary to prevent ollama from consuming all of the "while read line" input
# The 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' is to remove ollama's animation control characters from the output
	ollama run "${MODEL_NAME}" "${INPUT_STRING}" --nowordwrap </dev/null 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | tee -a "${output_file_name}" || true
}

while read -r line
do
	test_input_string "${OUTPUT_DIR}" "${MODEL_NAME}" "${line}"
done<"${INPUT_FILE}"
