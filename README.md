## GPT-SoVITS-API-V3
GPT-SoVITS api for v3 version

### 步骤

#### 第一步 放入代码

将本工程的代码依照目录结构拷贝进 GPT-SoVITS 工程中

#### 第二步 下载V3模型

下载V3模型，放入config/tts_util_infer_v3.yaml指定的目录下，或者根据模型位置修改 tts_util_infer_v3.yaml文件

#### 第三步 启动api进程

运行

```bash
python my_api_v3.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_util_infer_v3.yaml
```
