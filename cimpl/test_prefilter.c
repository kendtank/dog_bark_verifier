#include "prefilter_c.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <windows.h>


#pragma pack(push,1)
typedef struct {
    char riff_id[4];
    uint32_t riff_size;
    char wave_id[4];
    char fmt_id[4];
    uint32_t fmt_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
} WAVHeader;
#pragma pack(pop)

// ---------------------- 读取 PCM16 WAV ----------------------
int load_wav_mono_16bit(const char *path, float **out_buf, uint32_t *out_len, uint32_t *sample_rate)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;

    WAVHeader hdr;
    if (fread(&hdr, sizeof(WAVHeader), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }

    if (hdr.audio_format != 1 || hdr.num_channels != 1 || hdr.bits_per_sample != 16) {
        fclose(fp);
        printf("Only support PCM16 mono WAV\n");
        return -1;
    }

    *sample_rate = hdr.sample_rate;

    // 找 data 块
    char chunk_id[4];
    uint32_t chunk_size;
    while (fread(chunk_id, 1, 4, fp) == 4) {
        fread(&chunk_size, sizeof(uint32_t), 1, fp);
        if (strncmp(chunk_id, "data", 4) == 0) break;
        fseek(fp, chunk_size, SEEK_CUR);
    }

    int16_t *pcm = (int16_t *)malloc(chunk_size);
    if (!pcm) { fclose(fp); return -1; }
    fread(pcm, 1, chunk_size, fp);
    fclose(fp);

    uint32_t num_samples = chunk_size / 2;
    float *buf = (float *)malloc(sizeof(float) * num_samples);
    if (!buf) { free(pcm); return -1; }

    for (uint32_t i = 0; i < num_samples; i++)
        buf[i] = pcm[i] / 32768.0f;

    free(pcm);
    *out_buf = buf;
    *out_len = num_samples;
    return 0;
}

// ---------------------- 保存 PCM16 WAV ----------------------
int save_wav_mono_16bit(const char *path, const float *buf, uint32_t len, uint32_t sample_rate)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;

    WAVHeader hdr;
    memcpy(hdr.riff_id, "RIFF", 4);
    memcpy(hdr.wave_id, "WAVE", 4);
    memcpy(hdr.fmt_id, "fmt ", 4);
    hdr.fmt_size = 16;
    hdr.audio_format = 1;
    hdr.num_channels = 1;
    hdr.sample_rate = sample_rate;
    hdr.bits_per_sample = 16;
    hdr.block_align = hdr.num_channels * hdr.bits_per_sample / 8;
    hdr.byte_rate = hdr.sample_rate * hdr.block_align;
    hdr.riff_size = 36 + len * 2;

    fwrite(&hdr, sizeof(WAVHeader), 1, fp);

    fwrite("data", 1, 4, fp);
    uint32_t data_size = len * 2;
    fwrite(&data_size, sizeof(uint32_t), 1, fp);

    for (uint32_t i = 0; i < len; i++) {
        int16_t v = (int16_t)(buf[i] * 32767.0f);
        fwrite(&v, sizeof(int16_t), 1, fp);
    }

    fclose(fp);
    return 0;
}

// ---------------------- 主程序 ----------------------
int main(int argc, char **argv)
{
    SetConsoleOutputCP(CP_UTF8);  // 设置控制台输出为 UTF-8
    if (argc < 2) {
        printf("Usage: %s <wav_file>\n", argv[0]);
        return -1;
    }

    const char *file_path = argv[1];

    float *audio_buf = NULL;
    uint32_t audio_len = 0;
    uint32_t sr = 0;

    if (load_wav_mono_16bit(file_path, &audio_buf, &audio_len, &sr) != 0) {
        printf("Failed to read WAV file\n");
        return -1;
    }


    // ------------------------------
    // 初始化 Prefilter
    // ------------------------------
    BarkPrefilter prefilter;
    PrefilterConfig config;
    config.sample_rate = sr;
    config.energy_threshold = 1e-6f;
    config.peak_prominence = 0.0f;
    config.min_peak_distance_samples = 0;
    config.fine_merge_gap_samples = (uint32_t)(0.06f * sr);
    config.extend_head_samples = 0;
    config.extend_tail_samples = 0;

    init_prefilter(&prefilter, &config);

    // ------------------------------
    // 处理音频
    // ------------------------------
    BarkCandidate candidates[MAX_CANDIDATES];
    uint32_t num_candidates = preprocess_for_bark_detection(
        &prefilter, audio_buf, audio_len, NULL, candidates, MAX_CANDIDATES);

    printf("检测到 %u 个候选片段:\n", num_candidates);
    for (uint32_t i = 0; i < num_candidates; i++) {
        printf("  [%0.3f - %0.3f] 秒, 峰值能量: %f\n",
               candidates[i].start_time, candidates[i].end_time, candidates[i].peak_energy);

        // ------------------------------
        // 裁剪并保存
        // ------------------------------
        uint32_t start_idx = candidates[i].start;
        uint32_t end_idx = candidates[i].end;
        if (end_idx > audio_len) end_idx = audio_len;
        char out_name[256];
        snprintf(out_name, 256, "bark_segment_%02u.wav", i+1);
        save_wav_mono_16bit(out_name, audio_buf + start_idx, end_idx - start_idx, sr);
    }

    // ------------------------------
    // 导出能量 CSV
    // ------------------------------
    FILE *fp = fopen("energy.csv", "w");
    if (fp) {
        float energy[MAX_ENERGY_FRAMES];
        uint32_t energy_len;
        compute_short_time_energy(audio_buf, audio_len, energy, &energy_len);
        fprintf(fp, "time_sec,energy\n");
        for (uint32_t i = 0; i < energy_len; i++)
            fprintf(fp, "%f,%f\n", i * (HOP_LENGTH / (float)sr), energy[i]);
        fclose(fp);
        printf("能量曲线已导出到 energy.csv\n");
    }

    free(audio_buf);
    return 0;
}

// gcc test_prefilter.c prefilter_c.c -o test_prefilter.exe -lm
// .\test_prefilter.exe D:\work\code\dog_bark_verifier\data\dog_braking_test_mono.wav

