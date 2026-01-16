import os
import json
import argparse

import numpy as np
import soundfile as sf
import onnxruntime as ort
from tqdm import tqdm
from scipy.signal import stft, resample
import numpy.polynomial.polynomial as poly

print("ONNX device:", ort.get_device())

# ===== DNSMOS 标定系数（保持原样） =====
COEFS_SIG = np.array([
    9.651228012789436761e-01,
    6.592637550310214145e-01,
    7.572372955623894730e-02,
])
COEFS_BAK = np.array([
    -3.733460011101781717e+00,
    2.700114234092929166e+00,
    -1.721332907340922813e-01,
])
COEFS_OVR = np.array([
    8.924546794696789354e-01,
    6.609981731940616223e-01,
    7.600269530243179694e-02,
])


def audio_logpowspec(audio, nfft=320, hop_length=160, sr=16000):
    """
    使用 scipy.signal.stft 计算 log power spectrogram，返回形状 [T, F]
    并显式在时间轴两端 pad nfft//2，以模拟 librosa.stft(..., center=True)
    这样 9 秒音频在 nfft=320, hop=160 时会得到 901 帧，匹配 DNSMOS onnx 的输入。
    """
    # 模拟 librosa center=True：两端各 pad nfft//2 个采样点
    pad = nfft // 2
    audio_padded = np.pad(audio, (pad, pad), mode="constant", constant_values=0.0)

    # nperseg = 窗长, noverlap = 窗长 - hop_length
    _, _, Zxx = stft(
        audio_padded,
        fs=sr,
        nperseg=nfft,
        noverlap=nfft - hop_length,
        nfft=nfft,
        boundary=None,
        padded=False,
    )
    powspec = np.abs(Zxx) ** 2
    logpowspec = np.log10(np.maximum(powspec, 10 ** (-12)))
    # 现在是 [F, T] -> 转成 [T, F]
    return logpowspec.T


def load_segments_from_vad_json(vad_json_path, nparts=1, idx=0):
    """
    从 VAD json 中读取所有段，并按 nparts/idx 切片，方便并行。
    返回：list[dict]，每个 dict 包含：
        seg_id / audio_id / path / start_sec / end_sec / duration_sec / seg_idx
    """
    with open(vad_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segs = []
    for item in data:
        audio_id = item["audio_id"]
        path = item["path"]
        for i, seg in enumerate(item["segments"]):
            seg_id = f"{audio_id}_seg{i:03d}"
            segs.append(
                {
                    "seg_id": seg_id,
                    "audio_id": audio_id,
                    "path": path,
                    "start_sec": float(seg["start_sec"]),
                    "end_sec": float(seg["end_sec"]),
                    "duration_sec": float(seg["duration_sec"]),
                    "seg_idx": i,
                }
            )

    # 切片给多任务用
    slice_len = len(segs) // nparts if nparts > 0 else len(segs)
    start = idx * slice_len
    end = (idx + 1) * slice_len if idx != nparts - 1 else len(segs)
    return segs[start:end]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vad-json",
        type=str,
        required=True,
        help="VAD 结果 json 路径，例如 librispeech_silero_vad_segments_mp.json",
    )
    parser.add_argument(
        "--np",
        type=int,
        default=1,
        help="总分片数（多进程/多机并行时用）",
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=0,
        help="当前分片 index，从 0 开始",
    )
    parser.add_argument(
        "--input-length",
        type=int,
        default=9,
        help="DNSMOS 模型窗口长度（秒），默认 9s",
    )
    parser.add_argument(
        "--min-dur",
        type=float,
        default=1.0,
        help="最短处理时长（秒），短于该值的段直接跳过",
    )
    parser.add_argument(
        "--save-home",
        type=str,
        default="./",
        help="结果 tsv 保存目录",
    )
    args = parser.parse_args()

    # 1) 载入 VAD 段列表（当前 shard）
    segments = load_segments_from_vad_json(
        args.vad_json, nparts=args.np, idx=args.idx
    )
    print(f"Total segments for this shard: {len(segments)}")

    # 2) 加载 ONNX 模型（从当前目录找 sig.onnx / bak_ovr.onnx）
    this_dir = os.path.dirname(os.path.abspath(__file__))
    sig_model_path = os.path.join(this_dir, "sig.onnx")
    bak_ovr_model_path = os.path.join(this_dir, "bak_ovr.onnx")

    if not os.path.exists(sig_model_path):
        raise FileNotFoundError(f"sig.onnx not found at {sig_model_path}")
    if not os.path.exists(bak_ovr_model_path):
        raise FileNotFoundError(f"bak_ovr.onnx not found at {bak_ovr_model_path}")

    session_sig = ort.InferenceSession(
        sig_model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    session_bak_ovr = ort.InferenceSession(
        bak_ovr_model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    print("SIG providers:", session_sig.get_providers())
    print("BAK_OVR providers:", session_bak_ovr.get_providers())

    # 3) 输出文件
    os.makedirs(args.save_home, exist_ok=True)
    out_path = os.path.join(
        args.save_home, f"dns_from_vad_{args.idx}_{args.np}.tsv"
    )
    print("Saving result to:", out_path)

    with open(out_path, "w", encoding="utf-8") as wf:
        # 表头
        print(
            "seg_id\taudio_id\tpath\tstart_sec\tend_sec\tduration_sec\t"
            "mos_sig\tmos_bak\tmos_ovr",
            file=wf,
        )

        for seg in tqdm(segments, desc="DNSMOS scoring"):
            fpath = seg["path"]
            start_sec = seg["start_sec"]
            end_sec = seg["end_sec"]
            duration_sec = seg["duration_sec"]

            if not os.path.exists(fpath):
                # 文件不存在就跳过
                # 也可以选择 print warning
                continue

            # 读整条原始音频
            audio, fs = sf.read(fpath)
            # 多通道的话转单通道
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            if fs != 16000:
                target_sr = 16000
                new_len = int(len(audio) * target_sr / fs)
                audio = resample(audio, new_len)
                fs = target_sr

            # 截取对应的 VAD 段
            s = int(start_sec * fs)
            e = int(end_sec * fs)
            audio_seg = audio[s:e]

            if len(audio_seg) < int(args.min_dur * fs):
                # 段太短就不打分
                continue

            # 确保长度 >= input_length * fs，短的就重复拼接
            len_samples = int(args.input_length * fs)
            while len(audio_seg) < len_samples:
                audio_seg = np.append(audio_seg, audio_seg)

            hop_len_samples = fs  # 1s hop
            num_hops = int(np.floor(len(audio_seg) / fs) - args.input_length) + 1
            if num_hops < 1:
                num_hops = 1

            mos_sig_list, mos_bak_list, mos_ovr_list = [], [], []

            for i in range(num_hops):
                seg_slice = audio_seg[
                    int(i * hop_len_samples) : int(
                        (i + args.input_length) * hop_len_samples
                    )
                ]
                # [T, F] -> [1, T, F]
                feat = audio_logpowspec(seg_slice, sr=fs).astype("float32")[np.newaxis, :, :]

                # SIG
                onnx_inputs_sig = {
                    inp.name: feat for inp in session_sig.get_inputs()
                }
                sig_out = session_sig.run(None, onnx_inputs_sig)[0]
                # 保险写法：不管是 scalar / [1] / [1,1,1]，都转成一维再取第一个
                mos_sig_raw = float(np.array(sig_out).ravel()[0])
                mos_sig = float(poly.polyval(mos_sig_raw, COEFS_SIG))


                # BAK & OVR
                onnx_inputs_bak = {
                    inp.name: feat for inp in session_bak_ovr.get_inputs()
                }
                bak_ovr_out = session_bak_ovr.run(None, onnx_inputs_bak)[0]
                # 形状 [1, 3] 或 [1, 1, 3]，这里按 [0] 再取
                bak_ovr_vec = bak_ovr_out[0]
                # 保险一点再 ravel 一下
                bak_ovr_vec = np.array(bak_ovr_vec).ravel()
                mos_bak_raw = float(bak_ovr_vec[1])
                mos_ovr_raw = float(bak_ovr_vec[2])

                mos_bak = float(poly.polyval(mos_bak_raw, COEFS_BAK))
                mos_ovr = float(poly.polyval(mos_ovr_raw, COEFS_OVR))

                mos_sig_list.append(mos_sig)
                mos_bak_list.append(mos_bak)
                mos_ovr_list.append(mos_ovr)

            mean_sig = float(np.mean(mos_sig_list))
            mean_bak = float(np.mean(mos_bak_list))
            mean_ovr = float(np.mean(mos_ovr_list))

            print(
                f"{seg['seg_id']}\t{seg['audio_id']}\t{fpath}\t"
                f"{start_sec:.2f}\t{end_sec:.2f}\t{duration_sec:.2f}\t"
                f"{mean_sig:.3f}\t{mean_bak:.3f}\t{mean_ovr:.3f}",
                file=wf,
            )

    print("Done. Result saved to:", out_path)


if __name__ == "__main__":
    main()
