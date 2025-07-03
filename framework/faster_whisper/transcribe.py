from faster_whisper import WhisperModel
import torch

# 初始化模型
model_id ="bella-top/bella-whisper-large-v3"
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    model = WhisperModel(model_id, device=device, compute_type="float16")
else:
    model = WhisperModel(model_id, device=device)
# 转录音频
audio_path = "../samples/sample1.wav"  # 替换为你的音频文件路径
segments, info = model.transcribe(audio_path,
                                 repetition_penalty=1.2,
                                 language="zh",
                                 beam_size=5,
                                 word_timestamps=True,
                                 condition_on_previous_text=False,
                                 chunk_length=10,
                                 vad_filter=False,
                                 vad_parameters=dict(
                                     min_silence_duration_ms=400,
                                     max_speech_duration_s=25,
                                     speech_pad_ms=400))

# 打印转录结果
full_text = ""
for segment in segments:
    full_text += segment.text

print(f"\nFull transcribed text: {full_text}")