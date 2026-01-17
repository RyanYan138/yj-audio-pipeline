# yj-audio-pipeline
语音数据处理管线
一个“一键式”音频数据处理与转写管线，用于把一批原始音频从切分( VAD )、客观质量打分( DNSMOS )、质量过滤、说话人一致性过滤、语种过滤到多卡 Whisper 转写串起来跑完，并把中间产物都落盘，便于复现实验与调参。

Pipeline 做了什么

整体步骤如下（和 run_all.bash 对齐）：

VAD（Silero VAD）：扫描输入目录下音频，切出语音片段，输出 segments.json

DNSMOS（Docker 多 GPU）：对 VAD 片段做 DNSMOS 质量打分，输出分片 TSV（shard）

Merge DNSMOS shards：按 VAD json 的顺序把 shard TSV 合并成一个总 TSV

DNSMOS 过滤：按时长 + 分位数/阈值过滤低质量片段

Speaker Consistency：基于说话人 embedding 做一致性过滤（带 sqlite cache）

Language ID 过滤：过滤出目标语种（如 en），并生成 Whisper 输入用的 segments json

Whisper 多卡 ASR：按 shard 切分 segments，多 GPU 并行转写，最后 merge 成一个总 JSON

VAD 这一步用的是 silero-vad 包里提供的 load_silero_vad / read_audio / get_speech_timestamps 这套接口。

目录结构（约定）

通常你仓库里会是类似这样（按你现在的脚本路径写）：

run_all.bash：一键总入口（强烈建议只改顶部配置区）


vad/

vad_pipeline.py：Silero VAD 多进程切分


dns_mos/

eval_dns_from_vad_json.py：对 segments.json 打 DNSMOS

merge_dns_by_json_order.py：按 json 顺序 merge shards

filter_dnsmos_tsv.py：按阈值/分位数过滤


spk/

speaker_consistency_filter.py：说话人一致性过滤 + cache

language_filter/

lang_id_filter.py：语种识别与过滤

tsv_to_segments_json.py：把过滤后的 TSV 转回 Whisper segments.json（你之前报错就是缺它）


asr/

whisper_ms_from_segments.py：Whisper 多卡分片转写

merge_whisper_shards.py：merge shard 输出

output/：最终/中间输出（可自定义）


依赖与环境

Linux + bash

NVIDIA GPU + 驱动（DNSMOS / Whisper / 部分过滤步骤会用到 GPU）

Docker（DNSMOS 通过 docker 跑）

多个 Conda 环境（你在脚本里用的是绝对路径，比如）

Huawei_Encoder_Vad

spk_consistency

asr_whisper

快速开始（只建议改配置区）
1）克隆仓库
git clone git@github.com:RyanYan138/yj-audio-pipeline.git
cd yj-audio-pipeline

2）编辑 run_all.bash 顶部配置区

你主要会改这些：

PROJECT_ROOT：项目根目录

VAD_INPUT_ROOT：输入音频根目录

VAD_MAX_FILES：调试时限制处理数量（空=全量；填数字=只跑前 N 个）

DNSMOS_GPUS=(...)、WHISPER_GPUS=(...)：指定用哪些 GPU

FORCE=1/0：是否强制覆盖重跑（你想覆盖就设 1）

DO_VAD / DO_DNSMOS / ...：想跳过某步就设 0

3）一键运行
bash run_all.bash


跑完你会得到一串关键产物（脚本最后也会打印）：

VAD_OUT_JSON

DNSMOS_MERGED_TSV

DNSMOS_FILTERED_TSV

SPEAKER_OUT_TSV

LANG_OUT_FILTERED

LANG_SEG_JSON

WHISPER_FINAL_JSON

常用操作
只跑某一步 / 跳过某一步

例如只想从 Speaker 开始跑：

DO_VAD=0

DO_DNSMOS=0

DO_DNSMOS_FILTER=0

DO_SPEAKER=1

后面保持 1

调试模式：只跑前 N 个文件

把 VAD_MAX_FILES=100（或更小）即可。空字符串表示全量。

这次遇到的两个关键坑
坑 1：DNSMOS merge 把 “merged 文件自己”也当成 shard 读进去了

你日志里出现了：

“Found shard files” 里包含了 dns_from_vad_all.json_order.tsv

这会导致 merge 行为异常（甚至读到 0 行），因为 merge 输入应该只包含 dns_from_vad_0_2.tsv / dns_from_vad_1_2.tsv ... 这类 shard 文件。

最简单的规避方式：把 merged 输出放到单独子目录，避免 glob/扫描时把它也匹配进去，例如：

DNSMOS_MERGED_DIR="${DNSMOS_SAVE_HOME}/merged"
DNSMOS_MERGED_TSV="${DNSMOS_MERGED_DIR}/dns_from_vad_all.json_order.tsv"


并在 ensure_dirs() 里加：

mkdir -p "${DNSMOS_MERGED_DIR}"


这样 merge 脚本扫 shard 文件时就不会“误把 merged 当输入”。

坑 2：集群/网络封了 SSH 22 端口，GitHub 只能走 443

你一开始 ssh -T git@github.com 报 Permission denied (publickey)，但配置成走 443 后成功了。

原因：很多学校/集群出口会限制 22 端口；GitHub 官方支持把 SSH 走到 ssh.github.com:443。

你现在这种写法就可以（重点是 Host 还是 github.com，所以测试依然用 ssh -T git@github.com）：

Host github.com
  HostName ssh.github.com
  User git
  Port 443
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes


你之前 ssh -T github.com-443 报错是正常的：你并没有配置 Host github.com-443 这个别名，所以它解析不了。

Git 使用小抄（含“删除文件没同步”的原因）

你遇到的现象：删除了 setup.bash，但提交后远端还在。

原因是你用了 git add .，它在你的 Git 版本里默认不会记录“删除”（你也看到了那段 warning）。


正确做法二选一：

方式 A（推荐）：用 git add -A（会包含新增/修改/删除）

git add -A
git commit -m "remove setup.bash"
git push


方式 B：显式告诉 git 这是删除

git rm setup.bash
git commit -m "remove setup.bash"
git push

许可证 / 说明

本仓库主要是工程脚本编排与数据处理 glue code；VAD/DNSMOS/Whisper 等模型与实现分别遵循各自项目的许可与使用规范（你在生产/发布前建议把各组件 license 统一核对一遍）。
