// sherpa-onnx/csrc/mtk/common/Log.h
//
// Copyright (c)  2024  Xiaomi Corporation
// Adaptor for MTK code to use sherpa-onnx logging

#ifndef SHERPA_ONNX_CSRC_MTK_COMMON_LOG_H_
#define SHERPA_ONNX_CSRC_MTK_COMMON_LOG_H_

#include <sstream>
#include "sherpa-onnx/csrc/mtk/macros.h"

// Map MTK's stream-style LOG macros to printf-style sherpa-onnx logging
// MTK uses: LOG(INFO) << "message"
// We convert to: SHERPA_ONNX_LOGI("message")

namespace mtk {
namespace neuropilot {

// Stream logger that collects message and logs at destruction
class LogStream {
 public:
  enum Level { INFO, WARNING, ERROR };

  explicit LogStream(Level level) : level_(level) {}

  ~LogStream() {
    std::string msg = stream_.str();
    switch (level_) {
      case INFO:
        SHERPA_ONNX_LOGI("%s", msg.c_str());
        break;
      case WARNING:
        SHERPA_ONNX_LOGW("%s", msg.c_str());
        break;
      case ERROR:
        SHERPA_ONNX_LOGE("%s", msg.c_str());
        break;
    }
  }

  template <typename T>
  LogStream& operator<<(const T& val) {
    stream_ << val;
    return *this;
  }

 private:
  Level level_;
  std::ostringstream stream_;
};

// LOG macro creates a temporary LogStream
#define LOG(level) ::mtk::neuropilot::LogStream(::mtk::neuropilot::LogStream::level)

}  // namespace neuropilot
}  // namespace mtk

// DCHECK macro (debug check) - compatible with stream-style logging
// MTK uses: DCHECK(condition) << "message"
// We use a do-while to properly handle the stream expression
#define DCHECK(condition)                                                   \
  if (condition) {                                                          \
  } else                                                                    \
    ::mtk::neuropilot::LogStream(::mtk::neuropilot::LogStream::ERROR)       \
        << "DCHECK failed: " #condition " "

#define DCHECK_EQ(a, b) DCHECK((a) == (b))
#define DCHECK_NE(a, b) DCHECK((a) != (b))
#define DCHECK_LT(a, b) DCHECK((a) < (b))
#define DCHECK_LE(a, b) DCHECK((a) <= (b))
#define DCHECK_GT(a, b) DCHECK((a) > (b))
#define DCHECK_GE(a, b) DCHECK((a) >= (b))

#endif  // SHERPA_ONNX_CSRC_MTK_COMMON_LOG_H_
