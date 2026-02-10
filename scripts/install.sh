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
ssh ableton@move.local "mkdir -p /data/UserData/move-anything/modules/audio_fx/nam/models"
scp -r dist/nam/* ableton@move.local:/data/UserData/move-anything/modules/audio_fx/nam/

# Install chain presets if they exist
if [ -d "src/patches" ] && [ "$(ls -A src/patches 2>/dev/null)" ]; then
    echo "Installing chain presets..."
    ssh ableton@move.local "mkdir -p /data/UserData/move-anything/patches"
    scp src/patches/*.json ableton@move.local:/data/UserData/move-anything/patches/
fi

# Set permissions so Module Store can update later
echo "Setting permissions..."
ssh ableton@move.local "chmod -R a+rw /data/UserData/move-anything/modules/audio_fx/nam"

echo ""
echo "=== Install Complete ==="
echo "Module installed to: /data/UserData/move-anything/modules/audio_fx/nam/"
echo ""
echo "Place .nam model files in: /data/UserData/move-anything/modules/audio_fx/nam/models/"
echo "Restart Move Anything to load the new module."
