import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "bella-top/bella-whisper-large-v3"

# 加载模型和处理器
model =AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

# 创建推理管道
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# 加载本地音频文件
audio_path = "IT0022W0273.wav"  # 替换为你的音频文件路径
audio, sample_rate = librosa.load(audio_path, sr=16000)  # 确保采样率为 16kHz

# 使用管道进行推理
result = pipe(audio)
print("Transcription:", result["text"])