# bella-whisper

bella-whisper is a series of variant models based on OpenAI Whisper, designed for accurate speech recognition transcription. By fine-tuning with thousands of hours of high-quality data, bella-whisper performs excellently in multiple benchmark tests, especially in the real estate agent domain.

## 📚 Table of Contents
- [Model List](#model-list)
- [Highlights](#highlights)
- [Evaluation and Benchmarks](#evaluation-and-benchmarks)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [FAQ](#faq)
- [Model Acquisition and Licensing](#model-acquisition-and-licensing)

## 🤖 Model List

| Model Name | Base Model | Training Time | Hugging Face Repository |
|---------|---------|---------|-------------------------|
| bella-whisper-large-v3-turbo | whisper-large-v3-turbo | Not Open | N/A |
| **bella-whisper-large-v3** | **whisper-large-v3** | **20241118** | **[Link](https://huggingface.co/bella-top/bella-whisper-large-v3)** |
| bella-whisper-medium | whisper-medium | Not Open | N/A |
| bella-whisper-small | whisper-small | Not Open | N/A |
| bella-whisper-tiny | whisper-tiny | Not Open | N/A |


## ✨ Highlights
- 🚀 **Stable Performance**: Stable performance in multiple general benchmark tests.
- ✏️ **Punctuation Optimization**: More accurate context-based punctuation completion.
- 💡 **Specific Scenario Optimization**: Specially optimized for the real estate agent domain.

## 📊 Evaluation and Benchmarks
Model performance evaluation is a key part of measuring model effectiveness. We adopt standard evaluation metrics and conduct rigorous benchmark tests on public and custom test sets.
bella-whisper has demonstrated its superiority in multiple standard Chinese speech recognition benchmarks.

**Inference Framework**
Inference code can be found in **transcribe.py**.

### Character Error Rate (CER)
For character-based languages like Chinese, this is the most important metric. The formula is: CER = (Substitutions S + Deletions D + Insertions I) / Total Actual Characters N. Lower CER indicates better model performance.

#### Important Notes
All evaluation data has a sampling rate of 16K and is not included in the training data. Before calculating CER, both the predicted text (hypothesis) and reference text (reference) should undergo the same text normalization process to ensure fair comparison. The specific rules are as follows:

**(1). Unit Unification Conversion**
| Unit Symbol | Chinese Name |
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

**(2). Number to Chinese Characters Conversion**

For example: 111 -> 一百一十一 (yī bǎi yī shí yī)

**(3). Removal of Punctuation and Spaces**

For a stricter comparison, we also remove all punctuation and spaces.

#### Evaluation Method
Use the [jiwer](https://github.com/jitsi/jiwer) library to calculate CER.

#### Evaluation Results
The table below shows the performance of the fine-tuned models in this project on some standard Chinese test sets and specific scenario test sets, compared with baseline models. Please note: the data shown is an example, actual results may vary with training data, hyperparameters, and evaluation details. Please refer to the latest results actually released by the project.

| Dataset | bella-whisper-large-v3-20241118 | whisper-large-v3 |
|:--------:|:--------:|:--------:|
| AISHELL-1 Test | 6.700% | 5.562% |
| AISHELL-2 Test | 6.854% | 5.009% |
| wenetspeech test_net | 10.908% | 9.474% |
| wenetspeech test_meeting | 11.469% | 18.916% |
| [Internal] General Test Set-1 | 7.287% | 21.811% |
| [Internal] General Test Set-2 | 13.219% | 19.887% |

#### Some Examples
| Audio | whisper-large-v3 | bella-whisper-large-v3-20241118 | Reference |
|:--------:|:--------:|:--------:| :--------:|
| samples/sample1.wav | 赛后主攻朱廷霍最有价值球员和最受欢迎球员 | 赛后主攻**朱婷**或最有价值球员和最受欢迎球员。 | 赛后主攻朱婷获最有价值球员和最受欢迎球员 |
| samples/sample2.wav | 我们度过了平安故事的一年 | 我们度过了平安**无事**的一年。 | 我们度过了平安无事的一年 |
| samples/sample3.wav | 而成是最重要的网络 | 而**城市**最重要的网络。 | 而城市最重要的网络 |


#### Dataset Description
| Dataset | Sampling Rate | Data Scale | Description |
|:--------:|:--------:|:--------:|:--------:|
| AISHELL-1 Test | 16K | 10 hours | Test portion of a public Chinese Mandarin reading speech dataset |
| AISHELL-2 Test | 16K | 4 hours | Test portion of a public Chinese Mandarin reading speech dataset. |
| wenetspeech test_net | 16K | 12.6 hours | Mandarin reading speech dataset released by the Center for Speech and Language Technologies (CSLT), Tsinghua University |
| wenetspeech test_meeting | 16K | 15.2 hours | Mandarin reading speech dataset released by the Center for Speech and Language Technologies (CSLT), Tsinghua University |
| [Internal] General Test Set-1 | 16K | 5.1 hours | General data collection in a real environment |
| [Internal] General Test Set-2 | 16K | 74.8 hours | General data collection in a real environment |

## 🔧 Installation and Setup

Follow these steps to set up and install the runtime environment. We provide the following inference methods for model installation and execution.

**Environment Preparation**
| Hardware Environment | Operating System | Verified Environment |
|:--------:|:--------:|:--------:|
| gpu | linux | NVIDIA H20 + cuda_12.4 + Ubuntu20.04 |

**Clone Repository**
```bash
https://github.com/LianjiaTech/bella-whisper.git
cd bella-whisper
```

**Comparison of Different Inference Methods**
| Inference Method | Model Loading Speed | Inference Speed | GPU Memory Usage |
|:--------:|:--------:|:--------:|:--------:|
| faster-whisper | 14.54 seconds | 374.53 seconds | 3.6G |
| vllm | 574.08 seconds | 494.05 seconds | 72.6G |
| huggingface transformers | 169.86 seconds | 696.03 seconds | 3.87G |

- Environment:
  - Hardware: NVIDIA H20(95GB) * 1
- Samples: Total sample count 617, total duration 30 minutes.
- Statistical method: Loop through each audio, perform inference, and calculate total duration.

### 1. Using faster-whisper for Inference
#### Create Python Environment
```bash
conda create -n bella-faster-whisper python=3.10
conda activate bella-faster-whisper
```
Install dependencies

```bash
pip install -r framework/faster-whisper/requirements.txt
```

#### Run Inference

```python
from faster_whisper import WhisperModel
import torch

# Initialize model
model_id ="bella-top/bella-whisper-large-v3"
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    model = WhisperModel(model_id, device=device, compute_type="float16")
else:
    model = WhisperModel(model_id, device=device)
# Transcribe audio
audio_path = "sample1.wav"  # Replace with your audio file path
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

# Print transcription results
full_text = ""
for segment in segments:
    full_text += segment.text

print(f"\nFull transcribed text: {full_text}")
```
### 2. Using vllm for Inference
#### Create Python Environment
```bash
conda create -n bella-vllm-whisper python=3.10
conda activate bella-vllm-whisper
```
Install dependencies

```bash
pip install -r framework/vllm/requirements.txt
```
#### Run Inference
```python
from vllm import LLM, SamplingParams
import librosa

# Create Whisper model instance
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

# Load local audio file
audio_path = "sample1.wav"  # Replace with your audio file path
audio, sample_rate = librosa.load(audio_path, sr=None)
    # Prepare input prompts
prompts = [
        {
            "prompt": "<|startoftranscript|>",
            "multi_modal_data": {
                "audio": (audio, sample_rate),
            },
        }
    ]

    # Set sampling parameters
sampling_params = SamplingParams(
        temperature=0,
        top_p=1.0,
        max_tokens=500,
    )

# Start inference
outputs = llm.generate(prompts, sampling_params)

# Print results
for output in outputs:
    generated_text = output.outputs[0].text
    print(f"Generated text: {generated_text!r}")
```
### 3. Using huggingface transformers for Inference
#### Create Python Environment
```bash
conda create -n bella-huggingface-whisper python=3.10
conda activate bella-huggingface-whisper
```
Install dependencies

```bash
pip install -r framework/huggingface/requirements.txt
```
#### Run Inference
```python
import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "bella-top/bella-whisper-large-v3"

# Load model and processor
model =AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

# Create inference pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# Load local audio file
audio_path = "sample1.wav"  # Replace with your audio file path
audio, sample_rate = librosa.load(audio_path, sr=16000)  # Ensure sampling rate is 16kHz

# Perform inference using the pipeline
result = pipe(audio)
print("Transcription:", result["text"])
```
## ❓ FAQ
None yet
## 🔒 Model Acquisition and Licensing
 bella-whisper models can be obtained through the Hugging Face Model Hub. Before use, please ensure you have a Hugging Face account and have accepted the model's usage license. You may need to use a Hugging Face access token to download the model.

The project code is open-sourced under the [MIT License](https://opensource.org/licenses/MIT).
