include_directories(${PROJECT_SOURCE_DIR})

if(NOT DEFINED ANDROID_ABI)
  if(NOT DEFINED ENV{JAVA_HOME})
    message(FATAL_ERROR "Please set the environment variable JAVA_HOME")
  endif()
  include_directories($ENV{JAVA_HOME}/include)
  include_directories($ENV{JAVA_HOME}/include/linux)
  include_directories($ENV{JAVA_HOME}/include/darwin)
  include_directories($ENV{JAVA_HOME}/include/win32)
endif()

set(sources
  audio-tagging.cc
  jni.cc
  keyword-spotter.cc
  offline-punctuation.cc
  offline-recognizer.cc
  offline-speech-denoiser.cc
  offline-stream.cc
  online-punctuation.cc
  online-recognizer.cc
  online-stream.cc
  speaker-embedding-extractor.cc
  speaker-embedding-manager.cc
  spoken-language-identification.cc
  version.cc
  voice-activity-detector.cc
  wave-reader.cc
  wave-writer.cc
)

if(SHERPA_ONNX_ENABLE_TTS)
  list(APPEND sources
    offline-tts.cc
  )
endif()

if(SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION)
  list(APPEND sources
    offline-speaker-diarization.cc
  )
endif()

add_library(sherpa-onnx-jni SHARED ${sources})

target_compile_definitions(sherpa-onnx-jni PRIVATE SHERPA_ONNX_BUILD_SHARED_LIBS=1)
target_compile_definitions(sherpa-onnx-jni PRIVATE SHERPA_ONNX_BUILD_MAIN_LIB=1)

target_link_libraries(sherpa-onnx-jni sherpa-onnx-core)
install(TARGETS sherpa-onnx-jni DESTINATION lib)

# ----------------------------------------------------------------------------
# libasr_jni.so — slim wrapper for com.xbh.usbcloneclient.MtkAsrWrapper.
# Built only when MTK backend is enabled.
# ----------------------------------------------------------------------------
if(SHERPA_ONNX_ENABLE_MTK)
  add_library(asr_jni SHARED mtk-asr-jni.cc)
  target_compile_definitions(asr_jni PRIVATE SHERPA_ONNX_BUILD_SHARED_LIBS=1)
  target_compile_definitions(asr_jni PRIVATE SHERPA_ONNX_BUILD_MAIN_LIB=1)
  target_link_libraries(asr_jni sherpa-onnx-core)
  install(TARGETS asr_jni DESTINATION lib)
endif()
