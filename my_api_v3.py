#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import traceback
import requests
import hashlib
from typing import Generator

import argparse
import subprocess
import wave
import signal
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import FastAPI, UploadFile, File
import uvicorn
from io import BytesIO
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from tools.CutUtil import CutUtil
from tools.TTSUtilV3 import TTSUtilV3
from pydub import AudioSegment
from time import time as ttime


os.environ["TOKENIZERS_PARALLELISM"] = "false"
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

parser = argparse.ArgumentParser(description="GPT-SoVITS api")
parser.add_argument("-c", "--tts_config", type=str, default="GPT_SoVITS/configs/tts_util_infer_v3.yaml", help="tts_infer路径")
parser.add_argument("-a", "--bind_addr", type=str, default="0.0.0.0", help="default: 127.0.0.1")
parser.add_argument("-p", "--port", type=int, default="9880", help="default: 9880")
args = parser.parse_args()
config_path = args.tts_config

port = args.port
host = args.bind_addr
argv = sys.argv

if config_path in [None, ""]:
    config_path = "GPT-SoVITS/configs/tts_util_infer_v3.yaml"

tts_util = TTSUtilV3(config_path)

audio_file_save_path = os.path.join(os.getcwd(), "saved_audio")
os.makedirs(audio_file_save_path, exist_ok=True)

APP = FastAPI()


class TTSRequest(BaseModel):
    text: str = None
    text_lang: str = None
    ref_audio_path: str = None
    aux_ref_audio_paths: list = None
    prompt_lang: str = None
    prompt_text: str = ""
    top_k: int = 5
    top_p: float = 1
    temperature: float = 1
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: bool = False
    parallel_infer: bool = True
    repetition_penalty: float = 1.35


def check_params(req: dict):
    text: str = req.get("text", "")
    text_lang: str = req.get("text_lang", "")
    ref_audio_path: str = req.get("ref_audio_path", "")
    streaming_mode: bool = req.get("streaming_mode", False)
    media_type: str = req.get("media_type", "wav")
    prompt_lang: str = req.get("prompt_lang", "")
    text_split_method: str = req.get("text_split_method", "cut5")
    if ref_audio_path in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "ref_audio_path is required"})
    if text in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "text is required"})
    if text_lang in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "text_lang is required"})
    elif text_lang.lower() not in tts_util.list_language:
        return JSONResponse(status_code=400, content={
            "message": f"text_lang: {text_lang} is not supported in version {tts_util.version}"})
    if prompt_lang in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "prompt_lang is required"})
    elif prompt_lang.lower() not in tts_util.list_language:
        return JSONResponse(status_code=400, content={
            "message": f"prompt_lang: {prompt_lang} is not supported in version {tts_util.version}"})
    if media_type not in ["wav", "raw", "mp3"]:
        return JSONResponse(status_code=400, content={"message": f"media_type: {media_type} is not supported"})
    if text_split_method not in CutUtil.cut_method_names:
        return JSONResponse(status_code=400,
                            content={"message": f"text_split_method:{text_split_method} is not supported"})

    return None


@APP.post("/tts")
async def tts_post_endpoint(request: TTSRequest):
    req = request.dict()
    return await tts_handle(req)


@APP.get("/tts")
async def tts_get_endpoint(
        text: str = None,
        text_lang: str = None,
        ref_audio_path: str = None,
        aux_ref_audio_paths: list = None,
        prompt_lang: str = None,
        prompt_text: str = "",
        top_k: int = 20,
        top_p: float = 1,
        temperature: float = 1,
        text_split_method: str = "cut0",
        batch_size: int = 1,
        batch_threshold: float = 0.75,
        split_bucket: bool = True,
        speed_factor: float = 1.0,
        fragment_interval: float = 0.3,
        seed: int = -1,
        media_type: str = "wav",
        streaming_mode: bool = False,
        parallel_infer: bool = True,
        repetition_penalty: float = 1.35
):
    req = {
        "text": text,
        "text_lang": text_lang.lower(),
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": aux_ref_audio_paths,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang.lower(),
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": text_split_method,
        "batch_size": int(batch_size),
        "batch_threshold": float(batch_threshold),
        "speed_factor": float(speed_factor),
        "split_bucket": split_bucket,
        "fragment_interval": fragment_interval,
        "seed": seed,
        "media_type": media_type,
        "streaming_mode": streaming_mode,
        "parallel_infer": parallel_infer,
        "repetition_penalty": float(repetition_penalty)
    }
    return await tts_handle(req)


def pack_audio(io_buffer: BytesIO, data: np.ndarray, rate: int, media_type: str):
    if media_type == "mp3":
        io_buffer = pack_mp3(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer


def pack_mp3(io_buffer: BytesIO, data: np.ndarray, rate: int):
    t1 = ttime()
    wav_buffer = BytesIO()
    sf.write(wav_buffer, data, samplerate=rate, format="WAV")
    wav_buffer.seek(0)  # 重置指针
    wav_audio = AudioSegment.from_file(wav_buffer, format="wav")
    mp3_buffer = BytesIO()
    wav_audio.export(mp3_buffer, format="mp3")
    mp3_buffer.seek(0)
    t2 = ttime()
    print(f"pack_mp3 used [{t2 - t1} s]")
    return mp3_buffer


def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format='wav')
    return io_buffer


def get_file_name(url, folder="."):
    content = url
    m2 = hashlib.md5()
    m2.update(content.encode('utf-8'))
    md5_code = m2.hexdigest()

    # 从URL中提取文件名和后缀
    if "?" in url:
        url_no_param = url.split("?")[0]
        filename = url_no_param.split("/")[-1]
    else:
        filename = url.split("/")[-1]
    name, extension = os.path.splitext(filename)

    # 构造新的文件名
    new_filename = f"{md5_code}{extension}"

    # 构造完整的文件路径
    file_path = os.path.join(folder, new_filename)

    if os.path.exists(file_path):
        return file_path, True
    else:
        return file_path, False


def download_audio(url, folder="."):
    file_path, is_exist = get_file_name(url, folder)
    if is_exist:
        return file_path

    # 发送请求，下载文件
    response = requests.get(url, stream=True)

    # 检查请求是否成功
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:  # 过滤掉保持连接的新块
                    f.write(chunk)
        print(f"文件已下载到 {file_path}")
        return file_path
    else:
        print("请求失败，状态码：", response.status_code)
        return None


async def tts_handle(req: dict):
    """
    Text to speech handler.

    Args:
        req (dict):
            {
                "text": "",                   # str.(required) text to be synthesized
                "text_lang: "",               # str.(required) language of the text to be synthesized
                "ref_audio_path": "",         # str.(required) reference audio path
                "aux_ref_audio_paths": [],    # list.(optional) auxiliary reference audio paths for multi-speaker synthesis
                "prompt_text": "",            # str.(optional) prompt text for the reference audio
                "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
                "top_k": 5,                   # int. top k sampling
                "top_p": 1,                   # float. top p sampling
                "temperature": 1,             # float. temperature for sampling
                "text_split_method": "cut5",  # str. text split method, see text_segmentation_method.py for details.
                "batch_size": 1,              # int. batch size for inference
                "batch_threshold": 0.75,      # float. threshold for batch splitting.
                "split_bucket: True,          # bool. whether to split the batch into multiple buckets.
                "speed_factor":1.0,           # float. control the speed of the synthesized audio.
                "fragment_interval":0.3,      # float. to control the interval of the audio fragment.
                "seed": -1,                   # int. random seed for reproducibility.
                "media_type": "wav",          # str. media type of the output audio, support "wav", "raw", "ogg", "aac".
                "streaming_mode": False,      # bool. whether to return a streaming response.
                "parallel_infer": True,       # bool.(optional) whether to use parallel inference.
                "repetition_penalty": 1.35    # float.(optional) repetition penalty for T2S model.
            }
    returns:
        StreamingResponse: audio stream response.
    """

    print(str(req))
    streaming_mode = req.get("streaming_mode", False)
    return_fragment = req.get("return_fragment", False)
    media_type = req.get("media_type", "wav")

    check_res = check_params(req)
    if check_res is not None:
        return check_res

    ref_audio_path = req.get("ref_audio_path", "")
    ref_audio_path = ref_audio_path.strip()
    if ref_audio_path.startswith("http"):
        file_path = download_audio(ref_audio_path, audio_file_save_path)
        req["ref_audio_path"] = file_path
    if streaming_mode or return_fragment:
        req["return_fragment"] = True
    try:
        tts_generator = tts_util.run(req)

        if streaming_mode:
            def streaming_generator(tts_generator: Generator, media_type: str):
                if media_type == "wav":
                    yield wave_header_chunk()
                    media_type = "raw"
                for sr, chunk in tts_generator:
                    yield pack_audio(BytesIO(), chunk, sr, media_type).getvalue()

            return StreamingResponse(streaming_generator(tts_generator, media_type, ), media_type=f"audio/{media_type}")

        else:
            sr, audio_data = next(tts_generator)
            audio_data = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
            return Response(audio_data, media_type=f"audio/{media_type}")
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"tts failed", "Exception": str(e)})


def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    # This will create a wave header then append the frame input
    # It should be first on a streaming wav file
    # Other frames better should not have it (else you will hear some artifacts each chunk start)
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()


############


if __name__ == "__main__":
    try:
        if host == 'None':  # 在调用时使用 -a None 参数，可以让api监听双栈
            host = None
        uvicorn.run(app=APP, host=host, port=port, workers=1)
    except Exception as e:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
