# bella-whisper

bella-whisper是一系列基于OpenAI Whisper的变体模型，为实现精确的语音识别转写而设计。通过采用数千小时的高质量数据进行微调训练，bella-whisper在多个基准测试中表现出色，特别是在房产经纪领域。

## 📚目录
- [模型列表](#模型列表)
- [亮点](#亮点)
- [评估与基准](#评估与基准)
- [安装与设置](#安装与设置)
- [使用方法](#使用方法)
- [常见问题解答](#常见问题解答)
- [模型获取与许可](#模型获取与许可)
## 🤖 模型列表

| 模型名称 | 基座模型 | 训练时间 | Hugging Face 仓库链接 |
|---------|---------|---------|-----------------------|
| bella-whisper-large-v3-turbo | whisper-large-v3-turbo | 未开放 | N/A |
| **bella-whisper-large-v3** | **whisper-large-v3** | **20241118** | **[链接](https://huggingface.co/bella-top/bella-whisper-large-v3)** |
| bella-whisper-medium | whisper-medium | 未开放 | N/A |
| bella-whisper-small | whisper-small | 未开放 | N/A |
| bella-whisper-tiny | whisper-tiny | 未开放 | N/A |


## ✨ 亮点
- 🚀 **性能稳定**：在多个通用基准测试中表现稳定。
- ✏️ **标点符号优化**：较准确的根据语境的标点补全。
- 💡 **特定场景优化**：房产经纪领域经过特别优化。
## 📊 评估与基准
模型的性能评估是衡量模型效果的关键环节。我们采用标准的评估指标，并在公开及自定义测试集上进行严格的基准测试，
bella-whisper在多个标准语音识别基准测试中展现了其优越性。
**推理框架**
推理代码见**transcribe.py**。

### 字错误率 (CER)
对于中文等基于字符的语言，这是最重要的指标。计算公式为：CER = (替换数S + 删除数D + 插入数I) / 总真实字数N。CER越低，模型性能越好。
#### 重要说明
所有测评数据采样率均为16K，且在训练数据中均不包含。在计算CER之前，预测文本 (hypothesis) 和参考文本 (reference) 都应经过相同的文本规范化处理，以确保比较的公平性，具体规则如下：
**(1). 单位统一转换**
| 单位符号 | 中文名称 |
|:--------:|:--------:|
| ml, mL   | 毫升     |
| kL       | 千升     |
| m³, m3   | 立方米   |
| cm³, cm3 | 立方厘米 |
| ㎡, m², m2 | 平方米   |
| km², km2 | 平方千米 |
| cm², cm2 | 平方厘米 |
| ℃, °C    | 摄氏度   |
| K        | 开尔文   |
| °F       | 华氏度   |
| mm       | 毫米     |
| cm       | 厘米     |
| m        | 米       |
| km       | 千米     |
| mg       | 毫克     |
| g        | 克       |
| kg       | 千克     |
| t        | 吨       |
| s        | 秒       |
| min      | 分钟     |
| h        | 小时     |
| W        | 瓦       |
| kW       | 千瓦     |
| MW       | 兆瓦     |
| V        | 伏       |
| Ω        | 欧姆     |
| Hz       | 赫兹     |
| kHz      | 千赫兹   |
| MHz      | 兆赫兹   |
| Pa       | 帕斯卡   |
| kPa      | 千帕     |
| MPa      | 兆帕     |

**(2). 数字转汉字**
例如：111 -> 一百一十一

**(3). 去除标点符号及空格**
为了更严格的对比，我们还会去除所有标点符号和空格。

#### 评估方法
使用 [jiwer](https://github.com/jitsi/jiwer) 库来计算CER。

#### 评估结果
以下表格展示了本项目微调模型在一些标准中文测试集和特定场景测试集上的性能表现，并与基线模型进行对比。请注意：所示数据为示例，实际结果会随训练数据、超参数、评估细节而变化，请以项目实际发布的最新结果为准。

| 数据集 |  bella-whisper-large-v3-20241118 | whisper-large-v3 | 
|:--------:|:--------:|:--------:|
| AISHELL-1 Test | 6.700% | 5.562% | 
| AISHELL-2 Test | 6.854% | 5.009% | 
| wenetspeech test_net | 10.908% | 9.474% |
| wenetspeech test_meeting | 11.469% | 18.916% | 
| [内部] 通用测评集-1 | 7.287% | 21.811% |
| [内部] 通用测评集-2  | 13.219% | 19.887% | 

#### 一些例子
| Audio       | whisper-large-v3 | bella-whisper-large-v3-20241118 | 基准 |
|:--------:|:--------:|:--------:| :--------:|
| samples/sample1.wav   | 赛后主攻朱廷霍最有价值球员和最受欢迎球员| 赛后主攻**朱婷**或最有价值球员和最受欢迎球员。 | 赛后主攻朱婷获最有价值球员和最受欢迎球员 |
| samples/sample2.wav   | 我们度过了平安故事的一年| 我们度过了平安**无事**的一年。 | 我们度过了平安无事的一年 |
| samples/sample3.wav   | 而成是最重要的网络| 而**城市**最重要的网络。 | 而城市最重要的网络 |


#### 数据集说明
| 数据集 | 采样率 | 数据规模 | 说明 |
|:--------:|:--------:|:--------:|:--------:|
| AISHELL-1 Test | 16K | 10小时 | 公开的中文普通话朗读语音数据集的测试部分 |
| AISHELL-2 Test | 16K | 4小时 | 公开的中文普通话朗读语音数据集的测试部分。|
| wenetspeech test_net | 16K | 12.6小时 | 由清华大学语音与语言技术中心 (CSLT) 发布的普通话朗读语音数据集|
| wenetspeech test_meeting | 16K | 15.2小时 | 由清华大学语音与语言技术中心 (CSLT) 发布的普通话朗读语音数据集|
| [内部] 通用测评集-1 | 16K | 5.1小时 | 真实环境下的通用数据集合 |
| [内部] 通用测评集-2  | 16K | 74.8小时 | 真实环境下的通用数据集合 |

## 🔧 安装与设置

按照以下步骤来设置和安装运行环境。我们提供如下几种推理方式来进行模型安装和运行。

**环境准备**
| 硬件环境 | 操作系统 | 已经验证环境 |
|:--------:|:--------:|:--------:|
| gpu | linux | NVIDIA H20 + cuda_12.4 + Ubuntu20.04 |

**克隆仓库**
```bash
https://github.com/LianjiaTech/bella-whisper.git
cd bella-whisper
```

**不同推理方式对比**
| 推理方式 | 模型加载速度 | 推理速度 | 显存占用 |
|:--------:|:--------:|:--------:|:--------:|
| faster-whisper | 14.54秒 | 374.53秒 | 3.6G |
| vllm | 574.08秒 | 494.05秒  | 72.6G |
| huggingface transformers | 169.86秒 | 696.03秒 | 3.87G |

- 环境：
  - 硬件：NVIDIA H20(95GB) * 1
- 样本：总样本数量617个，总共时常30分钟。
- 统计方法：循环取出每个音频，进行推理，计算总时长。

### 1. 使用faster-whisper推理
#### 创建 Python 环境
```bash
conda create -n bella-faster-whisper python=3.10
conda activate bella-faster-whisper
```
安装依赖

```bash
pip install -r framework/faster-whisper/requirements.txt
```

#### 运行推理

```python
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
audio_path = "sample1.wav"  # 替换为你的音频文件路径
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
```
### 2. 使用vllm推理
#### 创建 Python 环境
```bash
conda create -n bella-vllm-whisper python=3.10
conda activate bella-vllm-whisper
```
安装依赖

```bash
pip install -r framework/vllm/requirements.txt
```
#### 运行推理
```python
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
audio_path = "sample1.wav"  # 替换为你的音频文件路径
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
```
### 3. 使用huggingface transformers推理
#### 创建 Python 环境
```bash
conda create -n bella-huggingface-whisper python=3.10
conda activate bella-huggingface-whisper
```
安装依赖

```bash
pip install -r framework/huggingface/requirements.txt
```
#### 运行推理
```python
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
audio_path = "sample1.wav"  # 替换为你的音频文件路径
audio, sample_rate = librosa.load(audio_path, sr=16000)  # 确保采样率为 16kHz

# 使用管道进行推理
result = pipe(audio)
print("Transcription:", result["text"])
```
## ❓ 常见问题解答
暂无
## 🔒 模型获取与许可
 bella-whisper模型可以通过 Hugging Face Model Hub 获取。使用前请确保您拥有 Hugging Face 账户，并已接受模型的使用许可。您可能需要使用 Hugging Face 访问令牌来下载模型。

项目代码基于 [MIT 许可证](https://opensource.org/licenses/MIT) 开源。
