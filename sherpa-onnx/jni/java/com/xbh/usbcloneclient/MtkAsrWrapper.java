package com.xbh.usbcloneclient;

/**
 * JNI wrapper for streaming Zipformer ASR on MTK NPU (sherpa-onnx backend).
 *
 * Model assets expected under {@code modelDir}:
 * <pre>
 *   encoder.dla
 *   decoder_npu.dla
 *   joiner.dla
 *   decoder_embedding_weight.npy
 *   vocab.txt
 * </pre>
 *
 * Typical real-time mic flow:
 * <pre>
 *   long h = MtkAsrWrapper.init(
 *       "/data/local/tmp/mtk_asr/models",
 *       "/data/local/tmp/mtk_asr/hotwords.txt",   // "" to disable
 *       "modified_beam_search",                    // or "greedy_search"
 *       2.0f,                                      // hotwords boost (method=beam)
 *       4);                                        // max-active-paths
 *   if (h == 0) { // init failed }
 *
 *   // feed 16 kHz mono float samples as they arrive from the mic
 *   while (recording) {
 *       float[] chunk = readMic();                 // e.g. 100 ms ~ 1600 samples
 *       MtkAsrWrapper.acceptWaveform(h, chunk, 16000);
 *
 *       String partial = MtkAsrWrapper.getResult(h);
 *       if (MtkAsrWrapper.isEndpoint(h)) {
 *           String utterance = partial;            // final for this segment
 *           onUtterance(utterance);
 *           MtkAsrWrapper.reset(h);                // start next segment
 *       } else {
 *           onPartial(partial);
 *       }
 *   }
 *   MtkAsrWrapper.inputFinished(h);                // flush remainder
 *   onUtterance(MtkAsrWrapper.getResult(h));
 *   MtkAsrWrapper.release(h);
 * </pre>
 */
public final class MtkAsrWrapper {

    // Load sherpa-onnx-core (plus its transitive runtime deps) explicitly so a
    // missing native lib surfaces a clear error here rather than as a late
    // UnsatisfiedLinkError from the first native method call.
    static {
        String[] libs = {
            "c++_shared",   // C++ runtime
            "onnxruntime",  // ORT backend (CPU fallback for unsupported ops)
            "asr_jni"       // this wrapper's JNI entry points (statically links
                            // sherpa-onnx-core + MTK provider)
        };
        for (String name : libs) {
            try {
                System.loadLibrary(name);
            } catch (UnsatisfiedLinkError e) {
                android.util.Log.e("MtkAsrWrapper",
                    "Failed to load lib" + name + ".so — make sure it is in "
                    + "lib/arm64-v8a/ of the APK. cause: " + e.getMessage());
                throw e;
            }
        }
    }

    private MtkAsrWrapper() {}

    // ----------------------------------------------------------------------
    // Native entry points (implemented in sherpa-onnx/jni/mtk-asr-jni.cc)
    // ----------------------------------------------------------------------

    /**
     * Load models and create an ASR session.
     *
     * @param modelDir        Absolute directory holding the .dla / .npy / vocab.
     * @param hotwordsFile    Path to hotwords.txt (one word per line, optional
     *                        {@code :score} suffix). Pass {@code ""} to disable.
     * @param decodingMethod  {@code "greedy_search"} or {@code "modified_beam_search"}.
     *                        Hotwords only take effect with {@code modified_beam_search}.
     * @param hotwordsScore   Global boost added to every hotword token
     *                        (typical 1.5 – 3.0).
     * @param maxActivePaths  Beam width for modified_beam_search (typical 4).
     * @return Opaque handle &gt; 0 on success, or 0 on failure.
     */
    public static native long init(String modelDir,
                                   String hotwordsFile,
                                   String decodingMethod,
                                   float hotwordsScore,
                                   int maxActivePaths);

    /**
     * Feed a chunk of 16 kHz mono float audio samples (values in [-1, 1]).
     * Safe to call as often as the mic delivers new samples; internally the
     * recognizer buffers + decodes whatever is ready.
     */
    public static native void acceptWaveform(long handle,
                                             float[] samples,
                                             int sampleRate);

    /**
     * Tell the recognizer no more samples will be pushed. Triggers a final
     * decode over any remaining buffered frames. Call once before
     * {@link #getResult(long)} to flush the tail of a finite utterance.
     */
    public static native void inputFinished(long handle);

    /** Latest decoded text (partial or final, depending on endpoint state). */
    public static native String getResult(long handle);

    /** True when an endpoint (silence / utterance end) has been detected. */
    public static native boolean isEndpoint(long handle);

    /**
     * Clear decoder state so decoding resumes for the next utterance. Call
     * after {@link #isEndpoint(long)} returns true and you have consumed the
     * final text with {@link #getResult(long)}.
     */
    public static native void reset(long handle);

    /**
     * Drop the current stream and start a new one with per-session hotwords
     * (in addition to any hotwords file passed at init). Pass {@code ""} to
     * clear dynamic hotwords. Use when the active hotword set changes at
     * runtime (e.g. contact-name list updated).
     */
    public static native void restart(long handle, String hotwordsText);

    /** Free all native resources. The handle is invalid after this call. */
    public static native void release(long handle);
}
