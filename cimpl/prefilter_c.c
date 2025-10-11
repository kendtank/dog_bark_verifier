// ------------------------------------
// 狗吠声检测前置滤波模块（C 实现版）
// ------------------------------------

#include "prefilter_c.h"
#include <string.h>
#include <math.h>
#include <stdbool.h>



// ===================== 预定义滤波器系数 =====================
// sos: [b0, b1, b2, a0, a1, a2]
// 这些系数假设来自 Python 的 butter(order=2, [300,8000], btype='band', fs=16000, output='sos')
const float PRECOMPUTED_SOS[NUM_SOS][6] = {
    {0.206572f, 0.0f, -0.206572f, -0.369527f, 0.586855f, 0.0f},
    {1.0f, 2.0f, 1.0f, -1.14298f, 0.412802f, 0.0f},
    {0.939091f, -1.878182f, 0.939091f, -1.876894f, 0.879675f, 0.0f}
};

// ===================== 滤波器实现 Python sosfilt() 的手写实现版 =====================
void apply_bandpass_filter(float *audio, uint32_t len)
{
    // 每个二阶节都要依次作用在信号上
    for (size_t s = 0; s < NUM_SOS; s++) {
        BiquadFilter f = {PRECOMPUTED_SOS[s][0], PRECOMPUTED_SOS[s][1], PRECOMPUTED_SOS[s][2],
                          PRECOMPUTED_SOS[s][4], PRECOMPUTED_SOS[s][5], 0, 0};
        // 对每个样本点进行滤波计算 二阶 IIR 滤波器（biquad），计算公式是：y[n]=b0​x[n]+b1​x[n−1]+b2​x[n−2]−a1​y[n−1]−a2​y[n−2]
        for (uint32_t i = 0; i < len; i++) {
            float x = audio[i];
            // 实际执行IIR滤波公式
            float y = f.b0 * x + f.z1;
            f.z1 = f.b1 * x - f.a1 * y + f.z2;
            f.z2 = f.b2 * x - f.a2 * y;
            audio[i] = y;
        }
    }
}

// ===================== 短时能量 =====================
void compute_short_time_energy(const float *audio, uint32_t audio_len, float *energy, uint32_t *energy_len)
{
    // 不足一帧，无法计算能量
    if (audio_len < FRAME_LENGTH) {
        *energy_len = 0;
        return;
    }

    // 计算帧数
    uint32_t frames = (audio_len - FRAME_LENGTH) / HOP_LENGTH + 1;
    if (frames > MAX_ENERGY_FRAMES) frames = MAX_ENERGY_FRAMES;
    *energy_len = frames;

    // 遍历每一帧
    for (uint32_t i = 0; i < frames; i++) {
        float sum = 0.0f;
        uint32_t start = i * HOP_LENGTH;
        for (uint32_t j = 0; j < FRAME_LENGTH; j++) {
            float v = audio[start + j];
            sum += v * v;
        }
        energy[i] = sum / FRAME_LENGTH;  // 均方能量
    }
}

// ===================== 自动阈值调整 =====================
void auto_tune_threshold(float *energy, uint32_t energy_len, PrefilterConfig *config)
{
    if (!energy || energy_len == 0 || !config)
        return;

    float max_energy = 0.0f;
    for (uint32_t i = 0; i < energy_len; i++)
        if (energy[i] > max_energy) max_energy = energy[i];

    config->peak_prominence = (max_energy * 0.05f > 1e-7f) ? max_energy * 0.05f : 1e-7f;
    uint32_t distance_samples = (uint32_t)(0.05f * config->sample_rate / HOP_LENGTH);
    config->min_peak_distance_samples = distance_samples > 1 ? distance_samples : 1;
}


// ===================== 简化版 find_peaks =====================
uint32_t find_peaks_enhanced(const float *x, uint32_t len,
                             float height, float prominence,
                             uint32_t min_distance, uint32_t min_width,
                             uint32_t *peaks, uint32_t max_peaks)
{
    if (!x || len < 3) return 0;
    if (!peaks) return 0;

    uint32_t count = 0;
    uint32_t i = 1;

    while (i < len - 1) {
        bool is_peak = (x[i] > x[i - 1] && x[i] >= x[i + 1]);
        if (is_peak && x[i] >= height) {
            float left_drop = x[i] - x[i - 1];
            float right_drop = x[i] - x[i + 1];
            float p = left_drop < right_drop ? left_drop : right_drop;

            if (p >= prominence) {
                uint32_t left = i, right = i;
                while (left > 0 && x[left - 1] < x[left]) left--;
                while (right < len - 1 && x[right + 1] < x[right]) right++;
                uint32_t width = right - left + 1;
                if (width >= min_width) {
                    if (count < max_peaks) peaks[count++] = i;
                    i += min_distance;
                    continue;
                }
            }
        }
        i++;
    }
    return count;
}

// ===================== 候选检测 =====================
uint32_t detect_candidates(const float *energy, uint32_t energy_len,
                           const PrefilterConfig *config,
                           BarkCandidate *events, uint32_t max_events)
{
    if (!energy || !events || energy_len == 0) return 0;

    uint32_t event_count = 0;
    uint32_t peaks[MAX_PEAKS];

    uint32_t peak_count = find_peaks_enhanced(energy, energy_len,
                                              config->energy_threshold,
                                              config->peak_prominence,
                                              config->min_peak_distance_samples / HOP_LENGTH,
                                              2,
                                              peaks, MAX_PEAKS);

    for (uint32_t p = 0; p < peak_count && event_count < max_events; p++) {
        uint32_t pk = peaks[p];
        float peak_e = energy[pk];
        uint32_t start = pk > 5 ? (pk - 5) * HOP_LENGTH : 0;
        uint32_t end   = ((pk + 5) * HOP_LENGTH > config->sample_rate * 10) ? config->sample_rate * 10 : (pk + 5) * HOP_LENGTH;
        BarkCandidate ev = {start, end, (float)start/config->sample_rate, (float)end/config->sample_rate, peak_e};
        events[event_count++] = ev;
    }

    return event_count;
}

// ===================== 自动合并事件 =====================
uint32_t auto_tune_peaks(BarkCandidate *events, uint32_t event_count, const PrefilterConfig *config)
{
    if (!events || event_count == 0 || !config) return 0;

    BarkCandidate merged[MAX_CANDIDATES];
    uint32_t merged_count = 0;
    merged[0] = events[0];
    merged_count = 1;

    for (uint32_t i = 1; i < event_count; i++) {
        BarkCandidate *prev = &merged[merged_count - 1];
        BarkCandidate *curr = &events[i];
        if (curr->start <= prev->end + config->fine_merge_gap_samples) {
            prev->end = MAX(prev->end, curr->end);
            prev->end_time = (float)prev->end / config->sample_rate;
        } else {
            merged[merged_count++] = *curr;
            if (merged_count >= MAX_CANDIDATES) break;
        }
    }

    memcpy(events, merged, merged_count * sizeof(BarkCandidate));
    return merged_count;
}

// ===================== 初始化 =====================
void init_prefilter(BarkPrefilter *prefilter, PrefilterConfig *config)
{
    if (!prefilter || !config) return;
    prefilter->config = *config;
    memset(prefilter->prefilter_buffer, 0, sizeof(prefilter->prefilter_buffer));
}

// ===================== 主流程 =====================
uint32_t preprocess_for_bark_detection(BarkPrefilter *prefilter,
                                       const float *audio, uint32_t audio_len,
                                       float *audio_out,
                                       BarkCandidate *final_events, uint32_t max_events)
{
    if (!prefilter || !audio || !final_events) return 0;

    if (audio_len > sizeof(prefilter->prefilter_buffer)/sizeof(float))
        audio_len = sizeof(prefilter->prefilter_buffer)/sizeof(float);

    memcpy(prefilter->prefilter_buffer, audio, sizeof(float) * audio_len);
    float *buf = prefilter->prefilter_buffer;

    apply_bandpass_filter(buf, audio_len);

    float energy[MAX_ENERGY_FRAMES];
    uint32_t energy_len;
    compute_short_time_energy(buf, audio_len, energy, &energy_len);

    auto_tune_threshold(energy, energy_len, &prefilter->config);

    BarkCandidate events[MAX_CANDIDATES];
    uint32_t event_count = detect_candidates(energy, energy_len, &prefilter->config, events, MAX_CANDIDATES);

    uint32_t final_count = auto_tune_peaks(events, event_count, &prefilter->config);
    memcpy(final_events, events, final_count * sizeof(BarkCandidate));

    if (audio_out && audio_out != audio)
        memcpy(audio_out, buf, sizeof(float) * audio_len);

    return final_count;
}