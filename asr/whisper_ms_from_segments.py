import json
import argparse
from pathlib import Path

import torch
import soundfile as sf
import numpy as np
import traceback

from modelscope import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers import pipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--seg_json",
        type=str,
        required=True,
        help="Silero VAD 生成的 segments_mp.json",
    )
    p.add_argument(
        "--out_json",
        type=str,
        required=True,
        help="输出 JSON 路径",
    )
    p.add_argument(
        "--model_dir",
        type=str,
        default="/Work21/2025/yanjiahao/modelscope_cache/models/AI-ModelScope/whisper-large-v3",
        help="whisper-large-v3 本地目录",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="ASR 推理 batch size（越大越吃 GPU）",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="指定设备，如 cuda:0 / cuda:1 / cpu；默认自动选第一块可用 GPU",
    )
    p.add_argument(
        "--language",
        type=str,
        default=None,
        help="已知语种可指定 zh/en 等；默认 None=自动识别",
    )
    p.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="仅用于调试：只处理前 N 条 entry",
    )
    # 多卡时手动分片用：同一份 seg_json，起多个进程，各自处理一部分
    p.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="总分片数，多卡时= GPU 数",
    )
    p.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="当前进程负责的分片 id，范围 [0, num_shards-1]",
    )
    return p.parse_args()


def load_audio_16k_mono(path: str):
    """读取音频 -> float32 mono 16k numpy"""
    audio, sr = sf.read(path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import torchaudio
        import torch as T

        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(T.from_numpy(audio).float()).numpy()
        sr = 16000
    return audio.astype(np.float32), sr


def main():
    args = parse_args()

    # ===== 设备选择（支持指定 cuda_index）=====
    if args.device is not None:
        device = args.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32

    print(f"Loading whisper-large-v3 from: {args.model_dir}")
    print(f"Device: {device}, dtype: {torch_dtype}, batch_size={args.batch_size}")
    print(f"Shard: {args.shard_id}/{args.num_shards}")

    # ===== 加载模型 & processor =====
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_dir,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(args.model_dir)

    asr_pipe = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,  # 官方推荐
        batch_size=args.batch_size,
        torch_dtype=torch_dtype,
        device=device,
    )

    # ===== 读取 VAD 结果 =====
    seg_json_path = Path(args.seg_json)
    with open(seg_json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    if args.max_files is not None:
        items = items[: args.max_files]

    total = len(items)

    # 多卡/多进程时按 index 分片：i % num_shards == shard_id
    if args.num_shards > 1:
        shard_items = [
            item for i, item in enumerate(items) if i % args.num_shards == args.shard_id
        ]
    else:
        shard_items = items

    print(
        f"Loaded {total} entries, this shard will process {len(shard_items)} entries."
    )

    results = []

    # ===== 批量推理缓存 =====
    batch_inputs = []
    batch_metas = []

    def flush_batch():
        nonlocal batch_inputs, batch_metas, results
        if not batch_inputs:
            return

        try:
            if args.language:
                outs = asr_pipe(
                    batch_inputs,
                    generate_kwargs={"language": args.language, "task": "transcribe"},
                )
            else:
                outs = asr_pipe(batch_inputs)

            outs_list = [outs] if isinstance(outs, dict) else outs
            for out, meta in zip(outs_list, batch_metas):
                r = meta.copy()
                r["text"] = (out.get("text") or "").strip()
                results.append(r)

        except Exception as e:
            print(f"[WARN] batch ASR failed: {repr(e)}")
            print(traceback.format_exc())

            msg = str(e).lower()
            err_type = "oom" if ("out of memory" in msg or ("cuda" in msg and "memory" in msg)) else "asr_error"

            for meta in batch_metas:
                r = meta.copy()
                r["text"] = ""
                r["error"] = f"asr_failed:{err_type}"
                results.append(r)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        finally:
            batch_inputs = []
            batch_metas = []


    # ===== 主循环：逐文件、逐 segment，攒 batch 推理 =====
    for idx, item in enumerate(shard_items, start=1):
        audio_id = item.get("audio_id")
        audio_path = item.get("path") or item.get("source")
        if not audio_path:
            continue
        audio_path = str(audio_path)

        try:
            audio, sr = load_audio_16k_mono(audio_path)
        except Exception as e:
            print(f"[WARN] load audio failed: {audio_path}, {e}")
            results.append(
                {
                    "audio_id": audio_id,
                    "path": audio_path,
                    "error": f"load_audio_failed: {e}",
                }
            )
            continue

        segs = item.get("segments", [])
        for j, seg in enumerate(segs, start=1):
            start_s = float(seg["start_sec"])
            end_s = float(seg["end_sec"])
            start = int(start_s * sr)
            end = int(end_s * sr)
            if end <= start:
                continue

            seg_audio = audio[start:end]

            batch_inputs.append({"array": seg_audio, "sampling_rate": sr})
            segid = seg.get("seg_id") or j
            meta = {
                "audio_id": audio_id,
                "seg_id": segid,
                "path": audio_path,
                "start_sec": start_s,
                "end_sec": end_s,
                "duration_sec": float(seg.get("duration_sec", end_s - start_s)),

                # 原来就有的
                "lang": seg.get("lang"),
                "lang_prob": seg.get("lang_prob"),
            }

                # 关键：把你清洗 pipeline 里已有的字段也透传出来（有就带上）
            for k in [
                "mos_sig", "mos_bak", "mos_ovr",
                "spk_id", "sim_spk", "sim_file",
                
            ]:
                if k in seg:
                    meta[k] = seg.get(k)

            batch_metas.append(meta)



            if len(batch_inputs) >= args.batch_size:
                flush_batch()

        if idx % 50 == 0:
            print(
                f"[shard {args.shard_id}] processed {idx}/{len(shard_items)} entries"
            )

    # 把最后不满一个 batch 的也跑完
    flush_batch()

    # ===== 写结果 =====
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done. Wrote {len(results)} segments to {out_path}")


if __name__ == "__main__":
    main()


#配合两个脚本“SingleGPU_Run”(单卡) and "MultiGPU_Run"(多卡),至于参数需要用户自己修改