#!/bin/bash
# run_test.sh

# Obtener la ruta absoluta al directorio de este script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting general test"
echo "---- TEST 1: CONTROLMPP ----"

CONTROLMPP_BIN_DIR="$SCRIPT_DIR/../ControlMPP/bin"
BIN_NAME="TestMPP"

if [ -x "$CONTROLMPP_BIN_DIR/$BIN_NAME" ]; then
    echo "Launching $BIN_NAME..."
    "$CONTROLMPP_BIN_DIR/$BIN_NAME"
else
    echo "Error: $BIN_NAME not found or not executable"
    exit 1
fi

read -n 1 -s -r -p "Press any button to start next test..."
echo ""

echo "---- TEST 2: VIDEOCAPTURE ----"

VIDEOCAPTURE_PYTHON_FILE="$SCRIPT_DIR/../VideoCapture/VideoCapture.py"
SOURCE="camera"
FRAMERATE=30

if [ -f "$VIDEOCAPTURE_PYTHON_FILE" ]; then
    echo "Running VideoCapture script..."
    python3 "$VIDEOCAPTURE_PYTHON_FILE" \
        --framerate "$FRAMERATE" \
        --source "$SOURCE" \
        --test
else
    echo "Script not found: $VIDEOCAPTURE_PYTHON_FILE"
    exit 1
fi

read -n 1 -s -r -p "Press any button to finish test..."
echo ""
