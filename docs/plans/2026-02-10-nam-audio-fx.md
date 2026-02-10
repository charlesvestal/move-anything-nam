# Neural Amp Modeler (NAM) Audio FX Module

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a Move Anything audio FX module that runs Neural Amp Modeler models for guitar amp/effect emulation.

**Architecture:** Wrap the NeuralAudio C++ library (header-only deps: Eigen, RTNeural, nlohmann/json) in a Move `audio_fx_api_v2` plugin. Mono processing (sum stereo to mono, process through NAM, write back to both channels). Model file browser via `items_param`/`select_param` pattern (like SF2 soundfont browser). Cross-compiled for ARM64 via Docker with C++20.

**Tech Stack:** C++20, NeuralAudio library (git submodule), Eigen (NEON auto-vectorization), CMake for NeuralAudio lib build, gcc for final plugin link.

---

### Task 1: Initialize Repository and Fetch NeuralAudio

**Files:**
- Create: `move-anything-nam/` repo structure
- Create: `.gitignore`

**Step 1: Init git repo**
```bash
cd /Volumes/ExtFS/charlesvestal/github/move-everything-parent/move-anything-nam
git init
```

**Step 2: Add NeuralAudio as submodule**
```bash
git submodule add https://github.com/mikeoliphant/NeuralAudio.git deps/NeuralAudio
cd deps/NeuralAudio && git submodule update --init --recursive && cd ../..
```

**Step 3: Create .gitignore**
```
build/
dist/
*.o
*.so
```

**Step 4: Create directory structure**
```bash
mkdir -p src/dsp scripts .github/workflows src/patches src/models
```

**Step 5: Commit**
```bash
git add -A && git commit -m "init: repo structure with NeuralAudio submodule"
```

---

### Task 2: Write module.json

**Files:**
- Create: `src/module.json`

**Step 1: Create module.json with model browser hierarchy**

The module needs:
- `component_type: "audio_fx"` with `chainable: true`
- A root level with knobs for input_level and output_level
- A "models" sublevel using `items_param`/`select_param` for .nam file browsing
- chain_params for input_level (0-1, default 0.5) and output_level (0-1, default 0.5)

Pattern follows SF2's soundfont browser: root level shows current model + params, "Choose Model" navigates to model list.

**Step 2: Commit**
```bash
git add src/module.json && git commit -m "feat: add module.json with model browser hierarchy"
```

---

### Task 3: Write the DSP Plugin (nam_plugin.cpp)

**Files:**
- Create: `src/dsp/nam_plugin.cpp`
- Copy: `src/dsp/audio_fx_api_v1.h` (from main repo, includes both v1 and the types we need)
- Copy: `src/dsp/plugin_api_v1.h` (from main repo, for host_api_v1_t)

**Step 1: Copy API headers from main repo**
```bash
cp move-anything/src/host/plugin_api_v1.h move-anything-nam/src/dsp/
cp move-anything/src/host/audio_fx_api_v1.h move-anything-nam/src/dsp/
cp move-anything/src/host/audio_fx_api_v2.h move-anything-nam/src/dsp/
```

**Step 2: Write nam_plugin.cpp**

Core structure:
```cpp
// Instance struct
typedef struct {
    char module_dir[256];
    NeuralAudio::NeuralModel *model;  // Currently loaded model
    char model_path[512];             // Current model file path
    char model_name[128];             // Display name (filename without extension)
    float input_level;                // 0.0 - 1.0, maps to dB adjustment
    float output_level;               // 0.0 - 1.0, maps to dB adjustment
    float input_gain;                 // Linear gain computed from input_level
    float output_gain;                // Linear gain computed from output_level
    int model_count;                  // Number of .nam files found
    char model_names[256][128];       // Scanned model filenames
    char model_paths[256][512];       // Full paths to model files
    bool model_loading;               // True while loading on background thread
    float mono_in[128];               // Mono input buffer
    float mono_out[128];              // Mono output buffer
} nam_instance_t;

// Entry point: move_audio_fx_init_v2
// create_instance: allocate, scan models dir, load first model if available
// destroy_instance: delete model, free
// process_block: sum stereo->mono, apply input gain, model->Process(), apply output gain, write stereo
// set_param: model (index), input_level, output_level
// get_param: model, model_name, model_count, model_list (JSON array), input_level, output_level, ui_hierarchy
```

Key implementation details:
- Scan `module_dir/models/` for `.nam` files on create and when `model_list` is queried
- Model loading: call `NeuralAudio::NeuralModel::CreateFromFile()` - this is blocking (~1-3s) so process_block passes audio through unchanged while loading
- Audio conversion: int16 stereo interleaved -> float mono -> NAM Process -> float -> int16 stereo
- `SetDefaultMaxAudioBufferSize(128)` in init
- Process in a single 128-frame call (NAM handles internal sub-blocking)

**Step 3: Commit**
```bash
git add src/dsp/ && git commit -m "feat: NAM audio FX plugin implementing audio_fx_api_v2"
```

---

### Task 4: Write Build System

**Files:**
- Create: `scripts/Dockerfile`
- Create: `scripts/build.sh`
- Create: `scripts/install.sh`

**Step 1: Create Dockerfile**

Ubuntu 22.04 with g++-aarch64-linux-gnu (GCC 12, supports C++20) and cmake.

**Step 2: Create build.sh**

Two-phase build:
1. Build NeuralAudio as a static library using CMake cross-compilation
2. Compile nam_plugin.cpp and link against libNeuralAudio.a

CMake cross-compile for NeuralAudio:
```bash
cmake -S deps/NeuralAudio -B build/neuralaudio \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
    -DCMAKE_CXX_STANDARD=20 \
    -DBUILD_UTILS=OFF \
    -DWAVENET_FRAMES=128 \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON

cmake --build build/neuralaudio -j$(nproc)
```

Then compile the plugin:
```bash
aarch64-linux-gnu-g++ -Ofast -shared -fPIC \
    -std=c++20 \
    -march=armv8-a -mtune=cortex-a72 \
    -DNDEBUG \
    src/dsp/nam_plugin.cpp \
    -o build/nam.so \
    -Isrc/dsp \
    -Ideps/NeuralAudio/NeuralAudio \
    -Ideps/NeuralAudio/deps/eigen \
    -Ideps/NeuralAudio/deps/RTNeural \
    -Ideps/NeuralAudio/deps/json/single_include \
    -Ideps/NeuralAudio/deps/math_approx/include \
    -Lbuild/neuralaudio/NeuralAudio \
    -lNeuralAudio \
    -lm -lstdc++ -lpthread
```

Package to dist/nam/ and create tarball.

**Step 3: Create install.sh**

SSH deploy to Move device at `modules/audio_fx/nam/`. Also copy any bundled .nam models to `modules/audio_fx/nam/models/`.

**Step 4: Commit**
```bash
git add scripts/ && git commit -m "feat: build system with Docker cross-compilation for ARM64"
```

---

### Task 5: Create GitHub Release Workflow

**Files:**
- Create: `.github/workflows/release.yml`

Follow tapescam pattern: trigger on `v*` tags, verify version match, Docker build, upload tarball.

**Step 1: Create release.yml**

**Step 2: Commit**
```bash
git add .github/ && git commit -m "ci: add release workflow"
```

---

### Task 6: Build and Test

**Step 1: Run build**
```bash
cd move-anything-nam && ./scripts/build.sh
```

**Step 2: Verify output**
```bash
ls -la dist/nam/
file dist/nam/nam.so  # Should be ELF 64-bit ARM aarch64
```

**Step 3: Fix any build issues and re-commit**

---

### Task 7: Create Signal Chain Preset

**Files:**
- Create: `src/patches/nam-default.json`

A basic chain preset with NAM as audio FX, no synth (user provides input signal).

**Step 1: Create preset**

**Step 2: Commit**
```bash
git add src/patches/ && git commit -m "feat: add default signal chain preset"
```
