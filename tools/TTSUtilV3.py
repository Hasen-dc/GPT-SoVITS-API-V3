import torch
import os
import json
import re
import torchaudio
import traceback
import gc

from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import random
import librosa
import yaml
from feature_extractor import cnhubert
from text import chinese
from text.LangSegmenter import LangSegmenter
from typing import List, Tuple, Union
from tools.CutUtil import CutUtil
from BigVGAN import bigvgan

from GPT_SoVITS.module.models import SynthesizerTrn, SynthesizerTrnV3
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from time import time as ttime
from module.mel_processing import spectrogram_torch, spec_to_mel_torch
from tools.my_utils import load_audio


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


class TTSUtilV3(object):
    list_language_v1 = [
        "all_zh",  # 全部按中文识别
        "en",  # 全部按英文识别#######不变
        "all_ja",  # 全部按日文识别
        "zh",  # 按中英混合识别####不变
        "ja",  # 按日英混合识别####不变
        "auto",  # 多语种启动切分识别语种
    ]

    list_language_v2 = [
        "all_zh",  # 全部按中文识别
        "en",  # 全部按英文识别#######不变
        "all_ja",  # 全部按日文识别
        "all_yue",  # 全部按中文识别
        "all_ko",  # 全部按韩文识别
        "zh",  # 按中英混合识别####不变
        "ja",  # 按日英混合识别####不变
        "yue",  # 按粤英混合识别####不变
        "ko",  # 按韩英混合识别####不变
        "auto",  # 多语种启动切分识别语种
        "auto_yue",  # 多语种启动切分识别语种
    ]

    def __init__(self, configs: Union[dict, str]):
        self.configs_path = configs
        self.configs: dict = self._load_configs(self.configs_path)
        self.configs: dict = self.configs.get("default")
        print(self.configs)

        self.version = self.configs.get("version", "v2")
        self.is_half = self.configs.get("is_half", False) and torch.cuda.is_available()
        self.dtype = torch.float16 if self.is_half else torch.float32
        self.model_version = self.version
        self.list_language = TTSUtilV3.list_language_v1 if self.version == 'v1' else TTSUtilV3.list_language_v2

        if torch.cuda.is_available():
            device_name = "cuda"
        else:
            device_name = "cpu"

        self.device = torch.device(device_name)
        self.bert_tokenizer: AutoTokenizer = None
        self.bert_model: AutoModelForMaskedLM = None

        self.t2s_weights_path = self.configs.get("t2s_weights_path", None)
        self.vits_weights_path = self.configs.get("vits_weights_path", None)
        self.bert_base_path = self.configs.get("bert_base_path", None)
        self.cnhubert_base_path = self.configs.get("cnhubert_base_path", None)
        self.bigvgan_path = self.configs.get("bigvgan_path", None)

        self.t2s_model = None
        self.hz = 50
        self.max_sec = 0

        self.hps = None
        self.vq_model = None
        self.model = None
        self.ssl_model = None
        self.resample_transform_dict = {}

        self.spec_min = -12
        self.spec_max = 2

        self.prompt_cache: dict = {
            "ref_audio_path": None,
            "prompt_semantic": None,
            "refer_spec": [],
            "prompt_text": None,
            "prompt_lang": None,
            "phones": None,
            "bert_features": None,
            "norm_text": None,
            "aux_ref_audio_paths": [],
        }
        self.cache = {}

        self.init_model()
        pass

    def _load_configs(self, configs_path: str) -> dict:
        with open(configs_path, 'r') as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)

        return configs

    def init_model(self):
        self.init_bert(self.bert_base_path)
        self.init_ssl_model(self.cnhubert_base_path)
        self.change_gpt_weights(self.t2s_weights_path)
        self.change_sovits_weights(self.vits_weights_path)
        if self.model_version != "v3":
            self.model = None
        else:
            self.init_bigvgan(self.bigvgan_path)

    def init_ssl_model(self, cnhubert_base_path):
        cnhubert.cnhubert_base_path = cnhubert_base_path
        self.ssl_model = cnhubert.get_model()
        if self.is_half:
            self.ssl_model = self.ssl_model.half().to(self.device)
        else:
            self.ssl_model = self.ssl_model.to(self.device)

    def init_bigvgan(self, bigvgan_path):

        self.model = bigvgan.BigVGAN.from_pretrained(bigvgan_path, use_cuda_kernel=False)
        # if True, RuntimeError: Ninja is required to load C++ extensions
        # remove weight norm in the model and set to eval mode
        self.model.remove_weight_norm()
        self.model = self.model.eval()
        if self.is_half:
            self.model = self.model.half().to(self.device)
        else:
            self.model = self.model.to(self.device)

    def init_bert(self, bert_path):
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
        if self.is_half:
            self.bert_model = self.bert_model.half().to(self.device)
        else:
            self.bert_model = self.bert_model.to(self.device)

    def change_gpt_weights(self, gpt_path):
        self.hz = 50
        dict_s1 = torch.load(gpt_path, map_location="cpu")
        config = dict_s1["config"]
        self.max_sec = config["data"]["max_sec"]
        self.t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
        self.t2s_model.load_state_dict(dict_s1["weight"])
        if self.is_half:
            self.t2s_model = self.t2s_model.half()
        self.t2s_model = self.t2s_model.to(self.device)
        self.t2s_model.eval()

        with open("./weight.json") as f:
            data = f.read()
            data = json.loads(data)
            data["GPT"][self.version] = gpt_path
        with open("./weight.json", "w") as f:
            f.write(json.dumps(data))

    def change_sovits_weights(self, sovits_path):
        '''
            v1:about 82942KB
            half thr:82978KB
            v2:about 83014KB
            half thr:100MB
            v1base:103490KB
            half thr:103520KB
            v2base:103551KB
            v3:about 750MB

            ~82978K~100M~103420~700M
            v1-v2-v1base-v2base-v3
            version:
                symbols version and timebre_embedding version
            self.model_version:
                sovits is v1/2 (VITS) or v3 (shortcut CFM DiT)
        '''
        size = os.path.getsize(sovits_path)
        if size < 82978 * 1024:
            self.model_version = version = "v1"
        elif size < 100 * 1024 * 1024:
            self.model_version = version = "v2"
        elif size < 103520 * 1024:
            self.model_version = version = "v1"
        elif size < 700 * 1024 * 1024:
            self.model_version = version = "v2"
        else:
            self.version = "v2"
            self.model_version = "v3"

        print(f"version:{self.version}, model_version:{self.model_version}")
        dict_s2 = torch.load(sovits_path, map_location="cpu", weights_only=False)
        self.hps = dict_s2["config"]
        self.hps = DictToAttrRecursive(self.hps)
        self.hps.model.semantic_frame_rate = "25hz"
        if dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
            self.hps.model.version = "v1"
        else:
            self.hps.model.version = "v2"
        self.version = self.hps.model.version
        # print("sovits版本:",self.hps.model.version)
        if self.model_version != "v3":
            self.vq_model = SynthesizerTrn(
                self.hps.data.filter_length // 2 + 1,
                self.hps.train.segment_size // self.hps.data.hop_length,
                n_speakers=self.hps.data.n_speakers,
                **self.hps.model
            )
            self.model_version = self.version
        else:
            self.vq_model = SynthesizerTrnV3(
                self.hps.data.filter_length // 2 + 1,
                self.hps.train.segment_size // self.hps.data.hop_length,
                n_speakers=self.hps.data.n_speakers,
                **self.hps.model
            )
        if "pretrained" not in sovits_path:
            try:
                del self.vq_model.enc_q
            except:
                pass
        if self.is_half:
            self.vq_model = self.vq_model.half().to(self.device)
        else:
            self.vq_model = self.vq_model.to(self.device)
        self.vq_model.eval()
        print("loading sovits_%s" % self.model_version, self.vq_model.load_state_dict(dict_s2["weight"], strict=False))
        with open("./weight.json") as f:
            data = f.read()
            data = json.loads(data)
            data["SoVITS"][self.version] = sovits_path
        with open("./weight.json", "w") as f:
            f.write(json.dumps(data))

    def run(self, inputs: dict):
        if_freeze = False
        inp_refs = None
        sample_steps = 8
        sampling_rate = 24000
        try:
            print("=========================== start run ===========================")

            self.stop_flag: bool = False
            text: str = inputs.get("text", "")
            text_lang: str = inputs.get("text_lang", "")
            ref_audio_path: str = inputs.get("ref_audio_path", "")
            aux_ref_audio_paths: list = inputs.get("aux_ref_audio_paths", [])
            prompt_text: str = inputs.get("prompt_text", "")
            prompt_lang: str = inputs.get("prompt_lang", "")
            top_k: int = inputs.get("top_k", 5)
            top_p: float = inputs.get("top_p", 1)
            temperature: float = inputs.get("temperature", 1)
            text_split_method: str = inputs.get("text_split_method", "cut0")
            batch_size = inputs.get("batch_size", 1)
            batch_threshold = inputs.get("batch_threshold", 0.75)
            speed_factor = inputs.get("speed_factor", 1.0)
            split_bucket = inputs.get("split_bucket", True)
            return_fragment = inputs.get("return_fragment", False)
            fragment_interval = inputs.get("fragment_interval", 0.3)
            seed = inputs.get("seed", -1)
            seed = -1 if seed in ["", None] else seed
            actual_seed = seed if seed not in [-1, "", None] else random.randrange(1 << 32)
            print(f"actual_seed :{actual_seed}")
            parallel_infer = inputs.get("parallel_infer", True)
            repetition_penalty = inputs.get("repetition_penalty", 1.35)

            t = []
            ref_free = False
            if prompt_text is None or len(prompt_text) == 0:
                ref_free = True
            if self.model_version == "v3":
                ref_free = False  # s2v3暂不支持ref_free
            if parallel_infer:
                print("并行推理模式已开启")
                self.t2s_model.model.infer_panel = self.t2s_model.model.infer_panel_batch_infer
            else:
                print("并行推理模式已关闭")
                self.t2s_model.model.infer_panel = self.t2s_model.model.infer_panel_naive_batched

            if return_fragment:
                print("分段返回模式已开启")
                if split_bucket:
                    split_bucket = False
                    print("分段返回模式不支持分桶处理，已自动关闭分桶处理")

            if split_bucket and speed_factor == 1.0:
                print("分桶处理模式已开启")
            elif speed_factor != 1.0:
                print("语速调节不支持分桶处理，已自动关闭分桶处理")
                split_bucket = False
            else:
                print("分桶处理模式已关闭")

            if fragment_interval < 0.01:
                fragment_interval = 0.01
                print("分段间隔过小，已自动设置为0.01")

            no_prompt_text = False
            if prompt_text in [None, ""]:
                no_prompt_text = True

            assert text_lang in self.list_language
            if not no_prompt_text:
                assert prompt_lang in self.list_language

            if ref_audio_path in [None, ""] and \
                    ((self.prompt_cache["prompt_semantic"] is None) or (self.prompt_cache["refer_spec"] in [None, []])):
                raise ValueError(
                    "ref_audio_path cannot be empty, when the reference audio is not set using set_ref_audio()")
            t0 = ttime()

            assert text_lang in self.list_language
            if not no_prompt_text:
                assert prompt_lang in self.list_language

            if not ref_free:
                prompt_text = prompt_text.strip("\n")
                if prompt_text[-1] not in CutUtil.splits:
                    prompt_text += "。" if prompt_lang != "en" else "."
            text = text.strip("\n")
            # if (text[0] not in splits and len(get_first(text)) < 4): text = "。" + text if text_language != "en" else "." + text

            print(f"实际输入的目标文本:{text}")
            zero_wav = np.zeros(
                int(self.hps.data.sampling_rate * 0.3),
                dtype=np.float16 if self.is_half else np.float32,
            )
            if not ref_free:
                with torch.no_grad():
                    wav16k, sr = librosa.load(ref_audio_path, sr=16000)
                    print(f"ref_audio shape:{wav16k.shape[0]}")
                    if wav16k.shape[0] > 170000 or wav16k.shape[0] < 48000:
                        raise OSError("参考音频在3~10秒范围外，请更换！")
                    wav16k = torch.from_numpy(wav16k)
                    zero_wav_torch = torch.from_numpy(zero_wav)
                    if self.is_half:
                        wav16k = wav16k.half().to(self.device)
                        zero_wav_torch = zero_wav_torch.half().to(self.device)
                    else:
                        wav16k = wav16k.to(self.device)
                        zero_wav_torch = zero_wav_torch.to(self.device)
                    wav16k = torch.cat([wav16k, zero_wav_torch])
                    ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))[
                        "last_hidden_state"
                    ].transpose(
                        1, 2
                    )  # .float()
                    codes = self.vq_model.extract_latent(ssl_content)
                    prompt_semantic = codes[0, 0]
                    prompt = prompt_semantic.unsqueeze(0).to(self.device)

            t1 = ttime()
            t.append(t1 - t0)

            text = CutUtil.cut_text(text_split_method, text)
            while "\n\n" in text:
                text = text.replace("\n\n", "\n")
            print(f"实际输入的目标文本(切句后):{text}")
            texts = text.split("\n")
            texts = self.process_text(texts)
            texts = self.merge_short_text_in_array(texts, 5)
            audio_opt = []
            ###s2v3暂不支持ref_free
            if not ref_free:
                phones1, bert1, norm_text1 = self.get_phones_and_bert(prompt_text, prompt_lang, self.version)

            for i_text, text in enumerate(texts):
                # 解决输入目标文本的空行导致报错的问题
                if len(text.strip()) == 0:
                    continue
                if text[-1] not in CutUtil.splits:
                    text += "。" if text_lang != "en" else "."
                phones2, bert2, norm_text2 = self.get_phones_and_bert(text, text_lang, self.version)
                if not ref_free:
                    bert = torch.cat([bert1, bert2], 1)
                    all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(self.device).unsqueeze(0)
                else:
                    bert = bert2
                    all_phoneme_ids = torch.LongTensor(phones2).to(self.device).unsqueeze(0)

                bert = bert.to(self.device).unsqueeze(0)
                all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)

                t2 = ttime()
                cache_key="%s-%s-%s-%s-%s-%s-%s-%s"%(ref_audio_path,prompt_text,prompt_lang,text,text_lang,top_k,top_p,temperature)
                # print(cache.keys(),if_freeze)
                if i_text in self.cache and if_freeze:
                    pred_semantic = self.cache[i_text]
                else:
                    with torch.no_grad():
                        pred_semantic, idx = self.t2s_model.model.infer_panel(
                            all_phoneme_ids,
                            all_phoneme_len,
                            None if ref_free else prompt,
                            bert,
                            # prompt_phone_len=ph_offset,
                            top_k=top_k,
                            top_p=top_p,
                            temperature=temperature,
                            early_stop_num=self.hz * self.max_sec,
                        )
                        # pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
                        pred_semantic = [item[-idx:] for item, idx in zip(pred_semantic, idx)]
                        pred_semantic = torch.cat(pred_semantic).unsqueeze(0).unsqueeze(0).to(self.device)
                        self.cache[i_text] = pred_semantic
                t3 = ttime()
                ###v3不存在以下逻辑和inp_refs
                if self.model_version != "v3":
                    refers = []
                    if inp_refs:
                        for path in inp_refs:
                            try:
                                refer = self.get_spepc(self.hps, path.name).to(self.dtype).to(self.device)
                                refers.append(refer)
                            except:
                                traceback.print_exc()
                    if len(refers) == 0:
                        refers = [self.get_spepc(self.hps, ref_audio_path).to(self.dtype).to(self.device)]
                    audio = (
                        self.vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(self.device).unsqueeze(0),
                                             refers,
                                             speed=speed_factor).detach().cpu().numpy()[0, 0])
                else:
                    refer = self.get_spepc(self.hps, ref_audio_path).to(self.device).to(
                        self.dtype)   # 这里要重采样切到32k,因为src是24k的，没有单独的32k的src，所以不能改成2个路径
                    phoneme_ids0 = torch.LongTensor(phones1).to(self.device).unsqueeze(0)
                    phoneme_ids1 = torch.LongTensor(phones2).to(self.device).unsqueeze(0)
                    fea_ref, ge = self.vq_model.decode_encp(prompt.unsqueeze(0), phoneme_ids0, refer)
                    ref_audio, sr = torchaudio.load(ref_audio_path)
                    ref_audio = ref_audio.to(self.device).float()
                    if ref_audio.shape[0] == 2:
                        ref_audio = ref_audio.mean(0).unsqueeze(0)
                    if sr != sampling_rate:
                        ref_audio = self.resample(ref_audio, sr, sampling_rate)
                    mel2 = self.mel_fn(ref_audio.to(self.dtype), sampling_rate)
                    mel2 = self.norm_spec(mel2)
                    T_min = min(mel2.shape[2], fea_ref.shape[2])
                    mel2 = mel2[:, :, :T_min]
                    fea_ref = fea_ref[:, :, :T_min]
                    if T_min > 468:
                        mel2 = mel2[:, :, -468:]
                        fea_ref = fea_ref[:, :, -468:]
                        T_min = 468
                    chunk_len = 934 - T_min
                    fea_todo, ge = self.vq_model.decode_encp(pred_semantic, phoneme_ids1, refer, ge)
                    cfm_resss = []
                    idx = 0
                    while True:
                        fea_todo_chunk = fea_todo[:, :, idx:idx + chunk_len]
                        if fea_todo_chunk.shape[-1] == 0:
                            break
                        idx += chunk_len
                        fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)
                        cfm_res = self.vq_model.cfm.inference(fea, torch.LongTensor([fea.size(1)]).to(fea.device), mel2,
                                                              sample_steps, inference_cfg_rate=0)
                        cfm_res = cfm_res[:, :, mel2.shape[2]:]
                        mel2 = cfm_res[:, :, -T_min:]
                        fea_ref = fea_todo_chunk[:, :, -T_min:]
                        cfm_resss.append(cfm_res)
                    cmf_res = torch.cat(cfm_resss, 2)
                    cmf_res = self.denorm_spec(cmf_res)
                    if self.model is None:
                        self.init_bigvgan(self.bigvgan_path)
                    with torch.inference_mode():
                        wav_gen = self.model(cmf_res)
                        audio = wav_gen[0][0].cpu().detach().numpy()
                    max_audio = np.abs(audio).max()  # 简单防止16bit爆音
                    if max_audio > 1:
                        audio /= max_audio
                audio_opt.append(audio)
                audio_opt.append(zero_wav)
                t4 = ttime()
                t.extend([t2 - t1, t3 - t2, t4 - t3])
                t1 = ttime()
            print("%.3f\t%.3f\t%.3f\t%.3f" %
                  (t[0], sum(t[1::3]), sum(t[2::3]), sum(t[3::3]))
                  )
            sr = self.hps.data.sampling_rate if self.model_version != "v3" else sampling_rate
            yield sr, (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)
        except Exception as e:
            print("=====================================")
            traceback.print_exc()
            # 必须返回一个空音频, 否则会导致显存不释放。
            yield sampling_rate, np.zeros(int(sampling_rate), dtype=np.int16)
            # 重置模型, 否则会导致显存释放不完全。
            del self.t2s_model
            del self.vq_model
            self.t2s_model = None
            self.vq_model = None
            self.change_gpt_weights(self.t2s_weights_path)
            self.change_sovits_weights(self.vits_weights_path)
            raise e
        finally:
            self.empty_cache()

    def empty_cache(self):
        try:
            gc.collect()  # 触发gc的垃圾回收。避免内存一直增长。
            if "cuda" in str(self.device):
                torch.cuda.empty_cache()
            elif str(self.device) == "mps":
                torch.mps.empty_cache()
        except:
            pass

    def process_text(self, texts):
        _text = []
        if all(text in [None, " ", "\n", ""] for text in texts):
            raise ValueError("请输入有效文本")
        for text in texts:
            if text in [None, " ", ""]:
                pass
            else:
                _text.append(text)
        return _text

    def merge_short_text_in_array(self, texts, threshold):
        if len(texts) < 2:
            return texts
        result = []
        text = ""
        for ele in texts:
            text += ele
            if len(text) >= threshold:

                result.append(text)
                text = ""
        if len(text) > 0:
            if len(result) == 0:
                result.append(text)
            else:
                result[len(result) - 1] += text
        return result

    def get_phones_and_bert(self, text, language, version, final=False):
        if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
            language = language.replace("all_", "")
            if language == "en":
                formattext = text
            else:
                # 因无法区别中日韩文汉字,以用户输入为准
                formattext = text
            while "  " in formattext:
                formattext = formattext.replace("  ", " ")
            if language == "zh":
                if re.search(r'[A-Za-z]', formattext):
                    formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                    formattext = chinese.mix_text_normalize(formattext)
                    return self.get_phones_and_bert(formattext, "zh", version)
                else:
                    phones, word2ph, norm_text = self.clean_text_inf(formattext, language, version)
                    bert = self.get_bert_feature(norm_text, word2ph).to(self.device)
            elif language == "yue" and re.search(r'[A-Za-z]', formattext):
                formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return self.get_phones_and_bert(formattext, "yue", version)
            else:
                phones, word2ph, norm_text = self.clean_text_inf(formattext, language, version)
                bert = torch.zeros(
                    (1024, len(phones)),
                    dtype=torch.float16 if self.is_half == True else torch.float32,
                ).to(self.device)
        elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
            textlist = []
            langlist = []
            if language == "auto":
                for tmp in LangSegmenter.getTexts(text):
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            elif language == "auto_yue":
                for tmp in LangSegmenter.getTexts(text):
                    if tmp["lang"] == "zh":
                        tmp["lang"] = "yue"
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            else:
                for tmp in LangSegmenter.getTexts(text):
                    if tmp["lang"] == "en":
                        langlist.append(tmp["lang"])
                    else:
                        # 因无法区别中日韩文汉字,以用户输入为准
                        langlist.append(language)
                    textlist.append(tmp["text"])
            print(textlist)
            print(langlist)
            phones_list = []
            bert_list = []
            norm_text_list = []
            for i in range(len(textlist)):
                lang = langlist[i]
                phones, word2ph, norm_text = self.clean_text_inf(textlist[i], lang, version)
                bert = self.get_bert_inf(phones, word2ph, norm_text, lang)
                phones_list.append(phones)
                norm_text_list.append(norm_text)
                bert_list.append(bert)
            bert = torch.cat(bert_list, dim=1)
            phones = sum(phones_list, [])
            norm_text = ''.join(norm_text_list)

        if not final and len(phones) < 6:
            return self.get_phones_and_bert("." + text, language, version, final=True)

        return phones, bert.to(self.dtype), norm_text

    def clean_text_inf(self, text, language, version):
        phones, word2ph, norm_text = clean_text(text, language, version)
        phones = cleaned_text_to_sequence(phones, version)
        return phones, word2ph, norm_text

    def get_bert_inf(self, phones, word2ph, norm_text, language):
        language = language.replace("all_", "")
        if language == "zh":
            bert = self.get_bert_feature(norm_text, word2ph).to(self.device)  # .to(dtype)
        else:
            bert = torch.zeros((1024, len(phones)), dtype=self.dtype, ).to(self.device)

        return bert

    def get_bert_feature(self, text, word2ph):
        with torch.no_grad():
            inputs = self.bert_tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T

    def mel_spectrogram(self, y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
        spec = spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center)
        mel = spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax)
        return mel

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min

    def mel_fn(self, x, sampling_rate):
        mel_fn_args = {
            "n_fft": 1024,
            "win_size": 1024,
            "hop_size": 256,
            "num_mels": 100,
            "sampling_rate": sampling_rate,
            "fmin": 0,
            "fmax": None,
            "center": False
        }
        return self.mel_spectrogram(x, **mel_fn_args)

    def resample(self, audio_tensor, sr0, resample_sr):
        if sr0 not in self.resample_transform_dict:
            self.resample_transform_dict[sr0] = torchaudio.transforms.Resample(
                sr0, resample_sr
            ).to(self.device)
        return self.resample_transform_dict[sr0](audio_tensor)

    def get_spepc(self, hps, filename):
        audio = load_audio(filename, int(hps.data.sampling_rate))
        audio = torch.FloatTensor(audio)
        maxx = audio.abs().max()
        if maxx > 1:
            audio /= min(2, maxx)
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm,
            hps.data.filter_length,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            center=False,
        )
        return spec
