#!/bin/bash

# Configuration
BASE_DIR="/home/anon/PROJECTS/uniAI/source_code/data/year_2/UHV"
EXTRACT_CMD="/home/anon/PROJECTS/uniAI/.venv/bin/python /home/anon/PROJECTS/uniAI/source_code/extract"


echo "Starting sequential extraction for UHV..."

# 1x Syllabus
echo "Starting Syllabus extraction..."
$EXTRACT_CMD/extract_multimodal_syllabus.py --path "$BASE_DIR/syllabus"

# 1x PYQ
echo "Starting PYQ extraction..."
$EXTRACT_CMD/extract_multimodal_pyq.py --path "$BASE_DIR/pyqs"

# 5x Notes (Unit 1-5)
echo "Starting Notes extraction (Units 1-5)..."
for i in {1..5}; do
    echo "  -> Starting Unit $i notes..."
    $EXTRACT_CMD/extract_multimodal_notes.py --path "$BASE_DIR/notes/unit$i"
done

echo "✅ All extractions completed successfully!"
