import vosk
import pyaudio
import numpy as np
import ollama
import json
import wave
from piper.voice import PiperVoice
import sounddevice as sd
#import time
from opencc import OpenCC

# 將 Text 轉成語音輸出
def text_to_speech(text, output_wav="output.wav"):
    # 轉換文字為語音
    text = text.replace("，", "").replace("。", "").replace("！", "").replace("？", "").replace("；", "").replace("：", "").replace("*", "")
    with wave.open(output_wav, "wb") as wav_file:
        wav_file.setnchannels(1)  # 單聲道
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(22050)  # 取樣率
        voice.synthesize(text, wav_file)  # 產生語音

    # 用 `sounddevice` 播放 `.wav`
    with wave.open(output_wav, "rb") as wf:
        sample_rate = wf.getframerate()  # 取得採樣率
        audio_data = wf.readframes(wf.getnframes())  # 讀取所有音訊幀

    # 轉換成 NumPy 陣列並播放
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    #time.sleep(0.25)  # 等待緩衝區填滿
    #緩衝區 blocksize 要設成 2048，如果設太小如 1024，聲音會斷斷續續，出現 ALSA lib pcm.c:8570:(snd_pcm_recover) underrun occurred
    sd.play(audio_array, samplerate=sample_rate, blocksize=2048, latency='low')
    sd.wait()  # 等待播放完成

# 將簡體中文轉換為繁體中文
def convert_to_traditional_chinese(text):
    cc = OpenCC('s2t')  # Simplified to Traditional
    return cc.convert(text) 

# 關閉 Vosk debug output
vosk.SetLogLevel(-1)

# 載入 Vosk 模型
print("載入 Vosk 模型...")
VOSK_MODEL_PATH = "vosk_models/vosk-model-small-cn-0.22"
model = vosk.Model(VOSK_MODEL_PATH)
rec = vosk.KaldiRecognizer(model, 16000)

# 載入 Piper 模型
print("載入 Piper 模型...")
PIPER_MODEL_PATH = "piper_models/zh_CN-huayan-medium.onnx"
voice = PiperVoice.load(PIPER_MODEL_PATH)

# 載入 Ollama 模型
print("載入 Ollama 模型...")
OLLAMA_MODEL_NAME = "gemma:2b"
response = ollama.chat(model=OLLAMA_MODEL_NAME, messages=[{"role": "system", "content": "你是我的性感女僕，用繁體中文顯示，並帶著挑逗的語氣回答問題。"}], stream=False)
print(response)

# 初始化 PyAudio
audio = pyaudio.PyAudio()

# 播放歡迎語音
print("\n帥氣的主人好，我是你的女僕！\n")
text_to_speech("帥氣的主人好，我是你的女僕！")

# 主循環
while True:
    print("\n帥氣的主人請說...\n")
    text_to_speech("帥氣的主人請說...")
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
    stream.start_stream()

    while True:
        data = stream.read(4096, exception_on_overflow=False)
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            if result["text"] != "":
                result["text"] = convert_to_traditional_chinese(result["text"])
                break
        # else:
        #     partial_result = rec.PartialResult()
        #     print('2:', partial_result)
    stream.stop_stream()
    stream.close()
    
    print("\n你說:\n", result["text"])
    if "離開" in result["text"]:
        print("\n再見了，我帥氣的主人！")
        text_to_speech("再見了，我帥氣的主人！")
        break
    # Collect more text before streaming
    messages = [{"role": "user", "content": result["text"]}]
    response = ollama.chat(model=OLLAMA_MODEL_NAME, messages=messages, stream=True)
    print("\n女僕 回應:\n")
    full_response = ""
    for chunk in response:
        chunk['message']["content"] = convert_to_traditional_chinese(chunk['message']["content"])
        print(chunk['message']["content"], end='', flush=True)
        # Collect the response in chunks
        full_response += chunk['message']["content"]
        if "\n" in full_response or any(p in full_response for p in "，。！？；："):
            text_to_speech(full_response.strip())
            full_response = ""
    if full_response:
        print('\n')
        text_to_speech(full_response.strip())