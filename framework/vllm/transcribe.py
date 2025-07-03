from vllm import LLM, SamplingParams
import librosa

# 创建 Whisper 模型实例
llm = LLM(
    model="bella-top/bella-whisper-large-v3",
    max_model_len=448,
    max_num_seqs=256,
    kv_cache_dtype="auto",
    dtype="float16",
    task="transcription",
    gpu_memory_utilization = 0.85,
    enforce_eager = False
)

# 加载本地音频文件
audio_path = "IT0022W0273.wav"  # 替换为你的音频文件路径
audio, sample_rate = librosa.load(audio_path, sr=None)
    # 准备输入提示
prompts = [
        {
            "prompt": "<|startoftranscript|>",
            "multi_modal_data": {
                "audio": (audio, sample_rate),
            },
        }
    ]

    # 设置采样参数
sampling_params = SamplingParams(
        temperature=0,
        top_p=1.0,
        max_tokens=500,
    )

# 开始推理
outputs = llm.generate(prompts, sampling_params)

# 打印结果
for output in outputs:
    generated_text = output.outputs[0].text
    print(f"Generated text: {generated_text!r}")