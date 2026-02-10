/*
 * NAM Audio FX Plugin - Neural Amp Modeler for Move Anything
 *
 * Wraps the NeuralAudio library (MIT, by Mike Oliphant) to run .nam and
 * .aidax neural-network guitar-amp models as a Signal Chain audio effect.
 *
 * Dependencies (all header-only / static, permissive licenses):
 *   NeuralAudio  - MIT      - Mike Oliphant
 *   Eigen        - MPL2     - Eigen contributors
 *   RTNeural     - BSD-3    - Jatin Chowdhury
 *   math_approx  - BSD-3    - Jatin Chowdhury
 *   nlohmann/json- MIT      - Niels Lohmann
 *
 * Audio: 44100 Hz, 128 frames/block, stereo interleaved int16 in-place.
 * NAM models are mono - we sum L+R to mono, process, write back to both.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <dirent.h>
#include <algorithm>
#include <string>
#include <atomic>
#include <pthread.h>

/* NeuralAudio */
#include "NeuralAudio/NeuralModel.h"

/* Move Anything API */
extern "C" {
#include "plugin_api_v1.h"
}

/* ======================================================================== */

#define MAX_MODELS 256
#define MAX_NAME_LEN 128
#define MAX_PATH_LEN 512
#define FRAMES_PER_BLOCK 128

static const host_api_v1_t *g_host = nullptr;

static void plugin_log(const char *msg) {
    if (g_host && g_host->log) g_host->log(msg);
}

/* ======================================================================== */
/* Instance                                                                  */
/* ======================================================================== */

typedef struct {
    char module_dir[MAX_PATH_LEN];

    /* Model */
    NeuralAudio::NeuralModel *model;
    std::atomic<NeuralAudio::NeuralModel *> pending_model;  /* set by loader thread */
    std::atomic<bool> loading;
    char model_path[MAX_PATH_LEN];
    char model_name[MAX_NAME_LEN];

    /* Scanned model files */
    int model_count;
    char model_names[MAX_MODELS][MAX_NAME_LEN];
    char model_paths[MAX_MODELS][MAX_PATH_LEN];
    int current_model_index;

    /* Parameters */
    float input_level;   /* 0.0 - 1.0 knob position */
    float output_level;  /* 0.0 - 1.0 knob position */
    float input_gain;    /* linear gain */
    float output_gain;   /* linear gain */

    /* Audio buffers (avoid per-block allocation) */
    float mono_in[FRAMES_PER_BLOCK];
    float mono_out[FRAMES_PER_BLOCK];

} nam_instance_t;

/* ======================================================================== */
/* Helpers                                                                   */
/* ======================================================================== */

/* Map 0-1 knob to dB range (-24 to +12), then to linear gain */
static float knob_to_gain(float knob) {
    float db = -24.0f + knob * 36.0f;  /* 0 -> -24dB, 0.5 -> -6dB, 1.0 -> +12dB */
    return powf(10.0f, db / 20.0f);
}

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* Strip directory and extension from path to get display name */
static void path_to_name(const char *path, char *name, int name_len) {
    const char *slash = strrchr(path, '/');
    const char *base = slash ? slash + 1 : path;
    const char *dot = strrchr(base, '.');
    int len = dot ? (int)(dot - base) : (int)strlen(base);
    if (len >= name_len) len = name_len - 1;
    memcpy(name, base, len);
    name[len] = '\0';
}

/* Check if filename ends with .nam or .json or .aidax */
static bool is_model_file(const char *name) {
    const char *dot = strrchr(name, '.');
    if (!dot) return false;
    return (strcasecmp(dot, ".nam") == 0 ||
            strcasecmp(dot, ".json") == 0 ||
            strcasecmp(dot, ".aidax") == 0);
}

/* Scan models directory and populate instance model list */
static void scan_models(nam_instance_t *inst) {
    char models_dir[MAX_PATH_LEN];
    snprintf(models_dir, sizeof(models_dir), "%s/models", inst->module_dir);

    inst->model_count = 0;

    DIR *dir = opendir(models_dir);
    if (!dir) {
        char msg[MAX_PATH_LEN + 64];
        snprintf(msg, sizeof(msg), "NAM: no models directory at %s", models_dir);
        plugin_log(msg);
        return;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr && inst->model_count < MAX_MODELS) {
        if (entry->d_name[0] == '.') continue;
        if (!is_model_file(entry->d_name)) continue;

        int idx = inst->model_count;
        snprintf(inst->model_paths[idx], MAX_PATH_LEN, "%s/%s", models_dir, entry->d_name);
        path_to_name(entry->d_name, inst->model_names[idx], MAX_NAME_LEN);
        inst->model_count++;
    }
    closedir(dir);

    /* Sort alphabetically by name */
    for (int i = 0; i < inst->model_count - 1; i++) {
        for (int j = i + 1; j < inst->model_count; j++) {
            if (strcasecmp(inst->model_names[i], inst->model_names[j]) > 0) {
                char tmp_name[MAX_NAME_LEN], tmp_path[MAX_PATH_LEN];
                memcpy(tmp_name, inst->model_names[i], MAX_NAME_LEN);
                memcpy(tmp_path, inst->model_paths[i], MAX_PATH_LEN);
                memcpy(inst->model_names[i], inst->model_names[j], MAX_NAME_LEN);
                memcpy(inst->model_paths[i], inst->model_paths[j], MAX_PATH_LEN);
                memcpy(inst->model_names[j], tmp_name, MAX_NAME_LEN);
                memcpy(inst->model_paths[j], tmp_path, MAX_PATH_LEN);
            }
        }
    }

    char msg[128];
    snprintf(msg, sizeof(msg), "NAM: found %d model files", inst->model_count);
    plugin_log(msg);
}

/* Background model loader thread */
static void *model_loader_thread(void *arg) {
    nam_instance_t *inst = (nam_instance_t *)arg;

    char msg[MAX_PATH_LEN + 64];
    snprintf(msg, sizeof(msg), "NAM: loading model %s", inst->model_path);
    plugin_log(msg);

    NeuralAudio::NeuralModel *new_model =
        NeuralAudio::NeuralModel::CreateFromFile(inst->model_path);

    if (new_model) {
        snprintf(msg, sizeof(msg), "NAM: model loaded successfully (sample_rate=%.0f)",
                 new_model->GetSampleRate());
        plugin_log(msg);
    } else {
        snprintf(msg, sizeof(msg), "NAM: failed to load model %s", inst->model_path);
        plugin_log(msg);
    }

    inst->pending_model.store(new_model, std::memory_order_release);
    inst->loading.store(false, std::memory_order_release);

    return nullptr;
}

static void load_model_async(nam_instance_t *inst, const char *path) {
    if (inst->loading.load(std::memory_order_acquire)) {
        plugin_log("NAM: already loading a model, skipping");
        return;
    }

    strncpy(inst->model_path, path, MAX_PATH_LEN - 1);
    inst->model_path[MAX_PATH_LEN - 1] = '\0';
    path_to_name(path, inst->model_name, MAX_NAME_LEN);

    inst->loading.store(true, std::memory_order_release);

    pthread_t thread;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    pthread_create(&thread, &attr, model_loader_thread, inst);
    pthread_attr_destroy(&attr);
}

/* ======================================================================== */
/* audio_fx_api_v2 implementation                                            */
/* ======================================================================== */

#define AUDIO_FX_API_VERSION_2 2
#define AUDIO_FX_INIT_V2_SYMBOL "move_audio_fx_init_v2"

typedef struct audio_fx_api_v2 {
    uint32_t api_version;
    void* (*create_instance)(const char *module_dir, const char *config_json);
    void (*destroy_instance)(void *instance);
    void (*process_block)(void *instance, int16_t *audio_inout, int frames);
    void (*set_param)(void *instance, const char *key, const char *val);
    int (*get_param)(void *instance, const char *key, char *buf, int buf_len);
    void (*on_midi)(void *instance, const uint8_t *msg, int len, int source);
} audio_fx_api_v2_t;

typedef audio_fx_api_v2_t* (*audio_fx_init_v2_fn)(const host_api_v1_t *host);

/* --- create_instance --- */
static void* v2_create_instance(const char *module_dir, const char *config_json) {
    (void)config_json;
    plugin_log("NAM: creating instance");

    NeuralAudio::NeuralModel::SetDefaultMaxAudioBufferSize(FRAMES_PER_BLOCK);

    nam_instance_t *inst = (nam_instance_t *)calloc(1, sizeof(nam_instance_t));
    if (!inst) return nullptr;

    strncpy(inst->module_dir, module_dir, MAX_PATH_LEN - 1);
    inst->model = nullptr;
    inst->pending_model.store(nullptr);
    inst->loading.store(false);
    inst->current_model_index = -1;

    /* Defaults: input at 0.5 (-6dB), output at 0.5 (-6dB) */
    inst->input_level = 0.5f;
    inst->output_level = 0.5f;
    inst->input_gain = knob_to_gain(0.5f);
    inst->output_gain = knob_to_gain(0.5f);

    /* Scan for model files */
    scan_models(inst);

    /* Load first model if available */
    if (inst->model_count > 0) {
        inst->current_model_index = 0;
        load_model_async(inst, inst->model_paths[0]);
    }

    return inst;
}

/* --- destroy_instance --- */
static void v2_destroy_instance(void *instance) {
    nam_instance_t *inst = (nam_instance_t *)instance;
    if (!inst) return;

    /* Wait for any pending load */
    while (inst->loading.load(std::memory_order_acquire)) {
        struct timespec ts = {0, 10000000}; /* 10ms */
        nanosleep(&ts, nullptr);
    }

    /* Clean up pending model if never consumed */
    NeuralAudio::NeuralModel *pending = inst->pending_model.load(std::memory_order_acquire);
    if (pending) delete pending;

    if (inst->model) delete inst->model;

    free(inst);
    plugin_log("NAM: instance destroyed");
}

/* --- process_block --- */
static void v2_process_block(void *instance, int16_t *audio_inout, int frames) {
    nam_instance_t *inst = (nam_instance_t *)instance;
    if (!inst) return;

    /* Check for newly loaded model (lock-free swap) */
    NeuralAudio::NeuralModel *pending = inst->pending_model.load(std::memory_order_acquire);
    if (pending) {
        NeuralAudio::NeuralModel *old = inst->model;
        inst->model = pending;
        inst->pending_model.store(nullptr, std::memory_order_release);
        if (old) delete old;
    }

    /* No model loaded - pass through */
    if (!inst->model) return;

    int n = (frames > FRAMES_PER_BLOCK) ? FRAMES_PER_BLOCK : frames;

    /* Deinterleave stereo int16 -> mono float */
    float ig = inst->input_gain;
    for (int i = 0; i < n; i++) {
        float l = audio_inout[i * 2]     / 32768.0f;
        float r = audio_inout[i * 2 + 1] / 32768.0f;
        inst->mono_in[i] = (l + r) * 0.5f * ig;
    }

    /* Process through NAM */
    inst->model->Process(inst->mono_in, inst->mono_out, (size_t)n);

    /* Convert back to stereo int16 */
    float og = inst->output_gain;
    for (int i = 0; i < n; i++) {
        float s = clampf(inst->mono_out[i] * og, -1.0f, 1.0f);
        int16_t sample = (int16_t)(s * 32767.0f);
        audio_inout[i * 2]     = sample;
        audio_inout[i * 2 + 1] = sample;
    }
}

/* --- set_param --- */
static void v2_set_param(void *instance, const char *key, const char *val) {
    nam_instance_t *inst = (nam_instance_t *)instance;
    if (!inst || !key || !val) return;

    if (strcmp(key, "input_level") == 0) {
        inst->input_level = clampf(atof(val), 0.0f, 1.0f);
        inst->input_gain = knob_to_gain(inst->input_level);
    } else if (strcmp(key, "output_level") == 0) {
        inst->output_level = clampf(atof(val), 0.0f, 1.0f);
        inst->output_gain = knob_to_gain(inst->output_level);
    } else if (strcmp(key, "model_index") == 0) {
        int idx = atoi(val);
        if (idx >= 0 && idx < inst->model_count && idx != inst->current_model_index) {
            inst->current_model_index = idx;
            load_model_async(inst, inst->model_paths[idx]);
        }
    } else if (strcmp(key, "model") == 0) {
        /* Direct path load */
        load_model_async(inst, val);
    }
}

/* --- get_param --- */
static int v2_get_param(void *instance, const char *key, char *buf, int buf_len) {
    nam_instance_t *inst = (nam_instance_t *)instance;
    if (!inst || !key || !buf) return -1;

    if (strcmp(key, "input_level") == 0)
        return snprintf(buf, buf_len, "%.2f", inst->input_level);
    if (strcmp(key, "output_level") == 0)
        return snprintf(buf, buf_len, "%.2f", inst->output_level);
    if (strcmp(key, "model_name") == 0)
        return snprintf(buf, buf_len, "%s", inst->model_name[0] ? inst->model_name : "(none)");
    if (strcmp(key, "model_count") == 0)
        return snprintf(buf, buf_len, "%d", inst->model_count);
    if (strcmp(key, "model_index") == 0)
        return snprintf(buf, buf_len, "%d", inst->current_model_index);

    /* Dynamic model list for Shadow UI browser - rescan each time */
    if (strcmp(key, "model_list") == 0) {
        scan_models(inst);

        int written = 0;
        written += snprintf(buf + written, buf_len - written, "[");
        for (int i = 0; i < inst->model_count && written < buf_len - 10; i++) {
            if (i > 0) written += snprintf(buf + written, buf_len - written, ",");
            written += snprintf(buf + written, buf_len - written, "\"%s\"", inst->model_names[i]);
        }
        written += snprintf(buf + written, buf_len - written, "]");
        return written;
    }

    if (strcmp(key, "loading") == 0)
        return snprintf(buf, buf_len, "%d", inst->loading.load(std::memory_order_acquire) ? 1 : 0);

    /* ui_hierarchy - returned dynamically so model name is current */
    if (strcmp(key, "ui_hierarchy") == 0) {
        const char *hierarchy = "{"
            "\"modes\":null,"
            "\"levels\":{"
                "\"root\":{"
                    "\"label\":\"NAM\","
                    "\"children\":null,"
                    "\"knobs\":[\"input_level\",\"output_level\"],"
                    "\"params\":["
                        "{\"key\":\"input_level\",\"label\":\"Input\"},"
                        "{\"key\":\"output_level\",\"label\":\"Output\"},"
                        "{\"level\":\"models\",\"label\":\"Choose Model\"}"
                    "]"
                "},"
                "\"models\":{"
                    "\"label\":\"Model\","
                    "\"items_param\":\"model_list\","
                    "\"select_param\":\"model_index\","
                    "\"children\":null,"
                    "\"knobs\":[],"
                    "\"params\":[]"
                "}"
            "}"
        "}";
        return snprintf(buf, buf_len, "%s", hierarchy);
    }

    return -1;
}

/* ======================================================================== */
/* Entry point                                                               */
/* ======================================================================== */

static audio_fx_api_v2_t g_fx_api_v2;

extern "C" audio_fx_api_v2_t* move_audio_fx_init_v2(const host_api_v1_t *host) {
    g_host = host;

    memset(&g_fx_api_v2, 0, sizeof(g_fx_api_v2));
    g_fx_api_v2.api_version     = AUDIO_FX_API_VERSION_2;
    g_fx_api_v2.create_instance = v2_create_instance;
    g_fx_api_v2.destroy_instance = v2_destroy_instance;
    g_fx_api_v2.process_block   = v2_process_block;
    g_fx_api_v2.set_param       = v2_set_param;
    g_fx_api_v2.get_param       = v2_get_param;
    g_fx_api_v2.on_midi         = nullptr; /* No MIDI handling needed */

    plugin_log("NAM: audio FX plugin initialized (NeuralAudio by Mike Oliphant)");

    return &g_fx_api_v2;
}
