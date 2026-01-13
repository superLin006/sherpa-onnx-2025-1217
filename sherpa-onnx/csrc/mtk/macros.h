// sherpa-onnx/csrc/mtk/macros.h
//
// Copyright (c)  2024  Xiaomi Corporation
// Copyright (c)  2024  MediaTek Inc.

#ifndef SHERPA_ONNX_CSRC_MTK_MACROS_H_
#define SHERPA_ONNX_CSRC_MTK_MACROS_H_

#include "sherpa-onnx/csrc/macros.h"

// SHERPA_ONNX_LOGE is defined in macros.h
// Add LOGI and LOGW for info and warning levels

#if __ANDROID_API__ >= 8
#define SHERPA_ONNX_LOGI(...)                                              \
  do {                                                                     \
    __android_log_print(ANDROID_LOG_INFO, "sherpa-onnx", ##__VA_ARGS__);   \
  } while (0)

#define SHERPA_ONNX_LOGW(...)                                              \
  do {                                                                     \
    __android_log_print(ANDROID_LOG_WARN, "sherpa-onnx", ##__VA_ARGS__);   \
  } while (0)
#elif defined(__OHOS__)
#define SHERPA_ONNX_LOGI(...) OH_LOG_INFO(LOG_APP, ##__VA_ARGS__)
#define SHERPA_ONNX_LOGW(...) OH_LOG_WARN(LOG_APP, ##__VA_ARGS__)
#else
#define SHERPA_ONNX_LOGI(...)                        \
  do {                                               \
    fprintf(stdout, "[INFO] %s:%s:%d ", __FILE__, __func__, \
            static_cast<int>(__LINE__));             \
    fprintf(stdout, ##__VA_ARGS__);                  \
    fprintf(stdout, "\n");                           \
  } while (0)

#define SHERPA_ONNX_LOGW(...)                        \
  do {                                               \
    fprintf(stderr, "[WARN] %s:%s:%d ", __FILE__, __func__, \
            static_cast<int>(__LINE__));             \
    fprintf(stderr, ##__VA_ARGS__);                  \
    fprintf(stderr, "\n");                           \
  } while (0)
#endif

// MTK NeuroPilot error checking macro
#define SHERPA_ONNX_MTK_CHECK(ret, msg, ...)                              \
  do {                                                                    \
    if (ret != 0) {                                                       \
      SHERPA_ONNX_LOGE("MTK Neuron error code: %d", static_cast<int>(ret)); \
      SHERPA_ONNX_LOGE(msg, ##__VA_ARGS__);                               \
      SHERPA_ONNX_EXIT(-1);                                               \
    }                                                                     \
  } while (0)

#define SHERPA_ONNX_MTK_CHECK_EQ(x, y, msg, ...)                          \
  do {                                                                    \
    if ((x) != (y)) {                                                     \
      SHERPA_ONNX_LOGE("MTK check failed: %d != %d",                      \
                       static_cast<int>(x), static_cast<int>(y));         \
      SHERPA_ONNX_LOGE(msg, ##__VA_ARGS__);                               \
      SHERPA_ONNX_EXIT(-1);                                               \
    }                                                                     \
  } while (0)

#endif  // SHERPA_ONNX_CSRC_MTK_MACROS_H_
