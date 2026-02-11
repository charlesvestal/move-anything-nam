#!/bin/bash
# Install NAM module to Move
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$REPO_ROOT"

if [ ! -d "dist/nam" ]; then
    echo "Error: dist/nam not found. Run ./scripts/build.sh first."
    exit 1
fi

echo "=== Installing NAM Module ==="

# Deploy to Move - audio_fx subdirectory
echo "Copying module to Move..."
ssh ableton@move.local "mkdir -p /data/UserData/move-anything/modules/audio_fx/nam/models /data/UserData/move-anything/modules/audio_fx/nam/cabs"
scp -r dist/nam/* ableton@move.local:/data/UserData/move-anything/modules/audio_fx/nam/

# Install chain presets if they exist
if [ -d "src/patches" ] && [ "$(ls -A src/patches 2>/dev/null)" ]; then
    echo "Installing chain presets..."
    ssh ableton@move.local "mkdir -p /data/UserData/move-anything/patches"
    scp src/patches/*.json ableton@move.local:/data/UserData/move-anything/patches/
fi

# Ask user for .nam model files
echo ""
echo "NAM requires .nam model files to function."
echo "Free models available at: https://tonehunt.org and https://tone3000.com"
echo ""
read -p "Path to folder containing .nam files (or Enter to skip): " NAM_DIR

if [ -n "$NAM_DIR" ] && [ -d "$NAM_DIR" ]; then
    NAM_COUNT=$(find "$NAM_DIR" -maxdepth 1 -name "*.nam" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$NAM_COUNT" -gt 0 ]; then
        echo "Copying $NAM_COUNT .nam model(s) to Move..."
        scp "$NAM_DIR"/*.nam ableton@move.local:/data/UserData/move-anything/modules/audio_fx/nam/models/
    else
        echo "No .nam files found in $NAM_DIR, skipping."
    fi
elif [ -n "$NAM_DIR" ]; then
    echo "Directory not found: $NAM_DIR, skipping."
fi

# Ask user for cabinet IR files
read -p "Path to folder containing .wav cabinet IRs (or Enter to skip): " CAB_DIR

if [ -n "$CAB_DIR" ] && [ -d "$CAB_DIR" ]; then
    CAB_COUNT=$(find "$CAB_DIR" -maxdepth 1 -name "*.wav" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$CAB_COUNT" -gt 0 ]; then
        echo "Copying $CAB_COUNT cabinet IR(s) to Move..."
        scp "$CAB_DIR"/*.wav ableton@move.local:/data/UserData/move-anything/modules/audio_fx/nam/cabs/
    else
        echo "No .wav files found in $CAB_DIR, skipping."
    fi
elif [ -n "$CAB_DIR" ]; then
    echo "Directory not found: $CAB_DIR, skipping."
fi

# Set permissions so Module Store can update later
echo "Setting permissions..."
ssh ableton@move.local "chmod -R a+rw /data/UserData/move-anything/modules/audio_fx/nam"

echo ""
echo "=== Install Complete ==="
echo "Module installed to: /data/UserData/move-anything/modules/audio_fx/nam/"
echo ""
echo "To add more models later, copy .nam files to the models/ folder on Move:"
echo "  scp your_model.nam ableton@move.local:/data/UserData/move-anything/modules/audio_fx/nam/models/"
echo ""
echo "Free NAM models: https://tonehunt.org and https://tone3000.com"
echo ""
echo "Restart Move Anything to load the new module."
