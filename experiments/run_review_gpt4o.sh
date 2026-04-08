#!/bin/bash

# Script to run the SciFi-Review GPT-4o configuration

# Exit on error
set -e

# Directory navigation is relative to the script location
# Assuming this script is located in experiments/

echo "=================================================="
echo "Starting SciFi-Review Framework for GPT-4o"
echo "=================================================="

cd multi_agent

echo "Running SciFi-Review..."
python llm_creativity.py \
    -c agent_roles_review_gpt4o.json \
    -d ../../datasets/SciFi/scientific_writing.json \
    -r 3 \
    -t SciFi-Review

echo "SciFi-Review Execution Completed."
echo ""
