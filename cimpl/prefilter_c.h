/*
 * prefilter_c.h
 * 狗吠声纹前置初筛预处理模块
 * 根据python算法迁移实现
 */

// 如果没有定义过宏 PREFILTER_C_H
#ifndef PREFILTER_C_H

// 定义宏 PREFILTER_C_H
#define PREFILTER_C_H

#include <stdint.h>
#include <string.h>
#include <math.h>

// ===================== 安全宏定义 =====================
#ifndef MIN
#define MIN(a,b) ({ __typeof__(a) _a = (a); __typeof__(b) _b = (b); _a < _b ? _a : _b; })
#endif

#ifndef MAX
#define MAX(a,b) ({ __typeof__(a) _a = (a); __typeof__(b) _b = (b); _a > _b ? _a : _b; })
#endif



// ===================== 常量定义 =====================
#define FRAME_LENGTH      400
#define HOP_LENGTH        100
#define MAX_ENERGY_FRAMES 16000
#define MAX_CANDIDATES    128
#define MAX_PEAKS         128
#define NUM_SOS           3


// ===================== 滤波器结构体 =====================
typedef struct {
    float b0, b1, b2;
    float a1, a2;
    float z1, z2;
} BiquadFilter;


// ===================== 候选事件结构体 =====================
typedef struct {
    uint32_t start;
    uint32_t end;
    float start_time;
    float end_time;
    float peak_energy;   // 🔥 必须加上，否则 detect_candidates() 会出错
} BarkCandidate;


// ===================== 预处理配置结构体 =====================
typedef struct {
    uint32_t sample_rate;
    float energy_threshold;             // 能量阈值
    float peak_prominence;              // 峰值突出度要求
    uint32_t min_peak_distance_samples; // 最小峰间距（样本）
    uint32_t fine_merge_gap_samples;    // 微调合并间距
    uint32_t extend_head_samples;       // 向前扩展
    uint32_t extend_tail_samples;       // 向后扩展
} PrefilterConfig;


// ===================== 预滤波主结构 =====================
typedef struct {
    PrefilterConfig config;
    float prefilter_buffer[48000 * 10]; // 10秒缓存
} BarkPrefilter;


// ===================== 滤波器系数 =====================
extern const float PRECOMPUTED_SOS[NUM_SOS][6];


// ===================== 函数声明 =====================

// 滤波器
void apply_bandpass_filter(float *audio, uint32_t len);

// 短时能量
void compute_short_time_energy(const float *audio, uint32_t audio_len,
                               float *energy, uint32_t *energy_len);

// 峰值检测（简化版 find_peaks）
uint32_t find_peaks_enhanced(const float *x, uint32_t len,
                             float height, float prominence,
                             uint32_t min_distance, uint32_t min_width,
                             uint32_t *peaks, uint32_t max_peaks);

// 自动阈值调整
void auto_tune_threshold(float *energy, uint32_t energy_len, PrefilterConfig *config);

// 候选检测
uint32_t detect_candidates(const float *energy, uint32_t energy_len,
                           const PrefilterConfig *config,
                           BarkCandidate *events, uint32_t max_events);

// 合并事件
uint32_t auto_tune_peaks(BarkCandidate *events, uint32_t event_count, const PrefilterConfig *config);

// 初始化
void init_prefilter(BarkPrefilter *prefilter, PrefilterConfig *config);

// 主流程
uint32_t preprocess_for_bark_detection(BarkPrefilter *prefilter,
                                       const float *audio, uint32_t audio_len,
                                       float *audio_out,
                                       BarkCandidate *final_events, uint32_t max_events);

#endif // PREFILTER_C_H