#ifndef UTILS_H
#define UTILS_H

#include <stddef.h> // for size_t
#include <float.h>  // for FLT_MAX, DBL_MAX
#include <math.h>   // for fmin(), fmax()

#ifdef __cplusplus
extern "C" {
#endif

// -----------------------------------------------------------------------------
// 最小值 / 最大值实现
// -----------------------------------------------------------------------------

// 类型安全的内联函数版本（整数）
static inline int min_int(int a, int b) {
    return (a < b) ? a : b;
}

static inline int max_int(int a, int b) {
    return (a > b) ? a : b;
}

// 浮点版本（float / double）
static inline float min_float(float a, float b) {
    return fminf(a, b); // fminf 是 C 标准库的安全实现
}

static inline float max_float(float a, float b) {
    return fmaxf(a, b);
}

static inline double min_double(double a, double b) {
    return fmin(a, b);
}

static inline double max_double(double a, double b) {
    return fmax(a, b);
}

// 通用宏（会自动根据变量类型调用正确函数）
// 注意：需要 C11 或 GNU 扩展支持 typeof
#if defined(__GNUC__) || defined(__clang__)
#define MIN(a, b) \
    __extension__ ({ __typeof__(a) _a = (a); __typeof__(b) _b = (b); \
    _a < _b ? _a : _b; })

#define MAX(a, b) \
    __extension__ ({ __typeof__(a) _a = (a); __typeof__(b) _b = (b); \
    _a > _b ? _a : _b; })
#else
// 如果编译器不支持 typeof，就退回安全函数版本
#define MIN(a, b) min_int((a), (b))
#define MAX(a, b) max_int((a), (b))
#endif

#ifdef __cplusplus
}
#endif

#endif // UTILS_H
