
# README


# yj-audio-pipeline

一个用于语音数据清洗与转写的工程化 Pipeline。  
它可以把一批原始音频按以下顺序串起来执行：

1. VAD 切段  
2. DNSMOS 打分  
3. DNSMOS 过滤  
4. 语种过滤（LID）  
5. Whisper 多卡转写  

中间结果会落盘，方便复现实验、观察每一步输出以及后续调参。



## 这个仓库面向两类人

### 1. 普通使用者
如果你只是想**把流程跑起来**，不关心内部实现细节，也不准备自己改代码：

**只需要使用 `run_all4user.bash`。**

你不需要研究其他 `run_all*.bash` 文件，也不需要关心我本地调试时用过的其他运行脚本。  
那些脚本主要是为了我自己做实验、调参数和优化流程保留的，不建议普通使用者直接修改或使用。

### 2. 开发 / 调优使用者
如果你需要：
- 调整阈值
- 修改流程顺序
- 更换模型
- 研究中间脚本实现

那再去看仓库里的其他运行脚本和子目录代码。



## Pipeline 做了什么

完整流程如下：

1. **VAD（Silero VAD）**  
   扫描输入目录中的音频，切出语音片段，输出 `segments.json`

2. **DNSMOS（Docker 多 GPU）**  
   对 VAD 片段进行客观质量打分，输出多个 shard TSV

3. **Merge DNSMOS shards**  
   按 VAD JSON 的顺序，把 DNSMOS shard 合并成一个总 TSV

4. **DNSMOS 过滤**  
   按时长与阈值过滤掉低质量片段

5. **Language ID（LID）**  
   过滤出目标语言，并生成 Whisper 输入所需的 `segments json`

6. **Whisper 多卡 ASR**  
   对过滤后的片段做多卡转写，最后 merge 成总 JSON



## Quick Start（普通用户只看这一节）

### 第一步：找我拿两个 Docker 镜像

当前推荐的运行方式是 **Docker 双镜像方案**。  
你需要先拿到这两个镜像：

- `yj-pipeline-runtime:with-spk`
- `dnsmos_gpu:cuda118`

我提供的使用说明里也是先导入这两个镜像，再使用脚本运行。:contentReference[oaicite:2]{index=2}

导入方式示例：

```bash
gzip -dc /你的路径/yj-pipeline-runtime_with-spk.tar.gz | docker load
gzip -dc /你的路径/dnsmos_gpu_cuda118.tar.gz | docker load
````

导入完成后可以检查：

```bash
docker images | egrep "yj-pipeline-runtime|dnsmos_gpu"
```

---

### 第二步：拉代码

```bash
git clone git@github.com:RyanYan138/yj-audio-pipeline.git
cd yj-audio-pipeline
```

我之前给的使用说明里也是这一步：从 GitHub 拉代码后，在目录里找到运行脚本并修改配置区。

---

### 第三步：只修改 `run_all4user.bash` 顶部配置区

普通使用者只需要打开：

```bash
run_all4user.bash
```

然后只修改最上面的配置区。

你主要会改这些内容：

* 项目目录
* 数据目录
* 输出实验名
* GPU 编号
* 模型权重目录
* 镜像名（如果你导入后镜像名没变，一般不用改）

你给用户使用的脚本本质上就是：
把两个镜像名、GPU、Whisper 模型路径和 LID 模型路径都集中放在顶部配置区。现有脚本里这些关键配置本来就集中在前面，比如镜像名、GPU、Whisper 模型目录和 LID 模型目录。

例如你需要重点改这些：

```bash
PROJECT_ROOT="/你的项目目录"
WORK_MOUNT_ROOT="/Work21"
DATA_MOUNT_ROOT="/CDShare3"

DATASET_NAME="你的实验名字"
INPUT_ROOT="/你的输入音频目录"

RUNTIME_IMAGE="yj-pipeline-runtime:with-spk"
DNSMOS_DOCKER_IMAGE="dnsmos_gpu:cuda118"

PIPELINE_GPUS=(0)
DNSMOS_GPUS=(0)
LID_GPU=0

WHISPER_MODEL_DIR_HOST="/你的whisper模型目录"
LID_MODEL_DIR_HOST="/你的LID模型目录"
```

其中：

* Whisper 这一步本来就是通过 `--model_dir` 读取本地模型目录。
* LID 这一步会把 `LANG_MODEL_ID` 传给 `lang_id_filter.py`。

---

### 第四步：直接运行

改完配置后，直接执行：

```bash
bash run_all4user.bash
```

就可以按顺序跑完整流程。

---

## 普通用户需要改哪些地方

对普通用户来说，通常只需要改：

1. **项目目录**
2. **输入音频目录**
3. **实验输出名**
4. **GPU 编号**
5. **Whisper 模型目录**
6. **LID 模型目录**
7. **镜像名（如果导入后名字变了才改）**

我的使用说明文档里也明确提到：
用户主要就是改目录、数据位置和模型权重位置，镜像名字如果没变则不用改。

---

## 哪个脚本给普通用户用

### 推荐脚本

```bash
run_all4user.bash
```

这个脚本是给普通用户准备的“直接运行版”。

### 不推荐普通用户使用的脚本

其他 `run_all*.bash`、调试脚本、实验脚本，主要用于：

* 我自己本地调试
* 对不同阶段做参数优化
* 做消融实验
* 修改流程逻辑

如果你只是想部署并运行，不需要碰它们。

---

## 目录结构说明

仓库核心目录大致如下：

```text
vad/
dns_mos/
language_filter/
asr/
output/
run_all4user.bash
```

其中：

* `vad/`：VAD 切段
* `dns_mos/`：DNSMOS 打分、合并、过滤
* `language_filter/`：语种识别与过滤
* `asr/`：Whisper 转写与合并
* `output/`：输出目录
* `run_all4user.bash`：给普通用户直接跑的脚本

---

## 输出结果

脚本跑完后，会在输出目录下生成一系列中间文件和最终文件，例如：

* VAD 输出 JSON
* DNSMOS 合并后的 TSV
* DNSMOS 过滤结果
* 语言过滤后的 TSV
* 语言过滤后的 segments JSON
* Whisper 最终转写结果 JSON

LID 和 Whisper 这两步在脚本中分别对应：

* `lang_id_filter.py`
* `whisper_ms_from_segments.py` + `merge_whisper_shards.py`。 

---

## 运行环境要求

建议在以下环境运行：

* Linux
* NVIDIA GPU
* Docker
* 已正确配置的 GPU Docker 环境

当前普通用户方案是 **双镜像运行**，不要求用户自己手动配置 Conda 环境。镜像导入后，直接通过脚本调用即可。你给的使用文档也是基于“先导入镜像，再改配置，再运行”的流程。

---

## 常见问题

### 1. 我只想直接跑，不想研究代码

只看 `Quick Start`，只用 `run_all4user.bash`。

### 2. 其他 run_all 脚本是做什么的

它们主要是我自己做实验、调试、优化流程时用的，不是给普通用户直接部署用的。

### 3. 我需要自己配 Python 环境吗

普通用户方案不需要手动配完整环境，直接用我提供的 Docker 镜像即可。

### 4. 我需要改很多参数吗

不需要。普通用户通常只要改：

* 目录
* 数据位置
* 模型位置
* GPU
* 输出实验名

这也是我给出的使用说明中的核心要求。

---

## 说明

这个仓库既包含“普通用户部署运行”的入口，也保留了我自己做实验和优化时使用的工程脚本。
如果你的目标只是**部署并跑起来**，请优先使用：

```bash
run_all4user.bash
```

不要直接修改其他运行脚本，除非你知道自己在做什么。

```

---


