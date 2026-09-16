#!/usr/bin/env python3
"""Batched Transformers inference for Fun-ASR-Nano without vLLM.

The upstream FunASRNano inference wrapper rejects batches before reaching the
model. This module keeps the upstream frontend and model weights, then batches
the expensive GPU stages manually:

    waveform -> fbank -> audio encoder/adaptor -> Qwen3 generate -> CTC align

FP16 is intentionally unsupported because this model's Qwen3 checkpoint
collapses to token 0. FP32 remains the default for NVIDIA V100 compatibility;
BF16 is optional on GPUs that support it natively.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


LOGGER = logging.getLogger("funasr_nano_batch")

AudioInput = Union[str, np.ndarray, torch.Tensor]
DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}
LANGUAGE_ALIASES = {
    "auto": None,
    "zh": "中文",
    "en": "英文",
    "ja": "日文",
}


class FunASRNanoBatch:
    """Run Fun-ASR-Nano with manual PyTorch/Transformers batching."""

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda:0",
        dtype: str = "fp32",
        max_new_tokens: int = 512,
    ) -> None:
        if dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {sorted(DTYPES)}, got {dtype!r}")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
        if (
            dtype == "bf16"
            and device.startswith("cuda")
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(f"BF16 is not supported by CUDA device: {device}")
        from funasr import AutoModel

        self.device = torch.device(device)
        self.dtype_name = dtype
        self.dtype = DTYPES[dtype]
        self.max_new_tokens = max_new_tokens

        LOGGER.info("Loading Fun-ASR-Nano from %s", model_dir)
        self.auto_model = AutoModel(
            model=model_dir,
            device=device,
            disable_update=True,
            disable_pbar=True,
            fp16=False,
            bf16=False,
            llm_conf={"llm_dtype": dtype},
        )
        self.model = self.auto_model.model
        self.tokenizer = self.auto_model.kwargs["tokenizer"]
        self.frontend = self.auto_model.kwargs["frontend"]

        # The upstream object caches llm_dtype from config.yaml during init.
        self.model.llm_dtype = dtype
        # SenseVoice/FSMN and CTC contain FP32-only paths in this FunASR build.
        # On BF16-capable GPUs, convert only Qwen, which is the main bottleneck.
        self.model.to(device=self.device)
        if dtype == "bf16":
            self.model.llm.to(dtype=self.dtype)
        self.model.eval()
        if dtype == "fp32":
            self._assert_no_bfloat16()

        self.pad_token_id = self.model.llm.config.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.model.llm.config.eos_token_id
        if self.pad_token_id is None:
            raise RuntimeError("Qwen3 config has neither pad_token_id nor eos_token_id")

        self.last_stats: Dict[str, float] = {}
        LOGGER.info(
            "Ready on %s with dtype=%s; no vLLM backend is used",
            self.device,
            self.dtype_name,
        )

    def _assert_no_bfloat16(self) -> None:
        bf16_tensors = [
            name
            for name, value in list(self.model.named_parameters())
            + list(self.model.named_buffers())
            if value.is_floating_point() and value.dtype == torch.bfloat16
        ]
        if bf16_tensors:
            preview = ", ".join(bf16_tensors[:5])
            raise RuntimeError(f"BF16 tensors remain in the model: {preview}")

    @staticmethod
    def _normalize_audio(audio: AudioInput) -> Union[str, torch.Tensor]:
        if isinstance(audio, np.ndarray):
            return torch.from_numpy(np.asarray(audio, dtype=np.float32)).squeeze()
        if isinstance(audio, torch.Tensor):
            return audio.detach().to(dtype=torch.float32, device="cpu").squeeze()
        if isinstance(audio, str):
            return audio
        raise TypeError(f"Unsupported audio input type: {type(audio).__name__}")

    def _prepare_one(
        self,
        audio: AudioInput,
        language: Optional[str],
        hotwords: Sequence[str],
        itn: bool,
    ) -> Dict[str, Any]:
        prompt = self.model.get_prompt(list(hotwords), language, itn)
        chatml = self.model.generate_chatml(prompt, self._normalize_audio(audio))
        contents = self.model.data_template(chatml)
        meta_data: Dict[str, Any] = {}
        runtime_kwargs = {
            key: value
            for key, value in self.auto_model.kwargs.items()
            if key not in {"tokenizer", "frontend"}
        }
        prepared = self.model.data_load_speech(
            contents,
            self.tokenizer,
            self.frontend,
            meta_data=meta_data,
            **runtime_kwargs,
        )
        if not isinstance(prepared["speech"], torch.Tensor) or prepared["speech"].numel() == 0:
            raise RuntimeError("Fun-ASR-Nano frontend returned no speech features")
        return {
            "contents": contents,
            "prepared": prepared,
            "meta_data": meta_data,
        }

    def _encode_audio_batch(self, samples: Sequence[Dict[str, Any]]):
        features = [sample["prepared"]["speech"][0] for sample in samples]
        feature_lengths = torch.tensor(
            [
                int(sample["prepared"]["speech_lengths"].reshape(-1)[0].item())
                for sample in samples
            ],
            dtype=torch.int32,
            device=self.device,
        )
        speech = pad_sequence(features, batch_first=True, padding_value=0.0)
        speech = speech.to(device=self.device, dtype=torch.float32)

        encoder_out, encoder_out_lens = self.model.encode(speech, feature_lengths)
        adaptor_out, adaptor_out_lens = self.model.audio_adaptor(
            encoder_out, encoder_out_lens
        )
        return encoder_out, encoder_out_lens, adaptor_out, adaptor_out_lens

    def _build_prompt_batch(
        self,
        samples: Sequence[Dict[str, Any]],
        adaptor_out: torch.Tensor,
        adaptor_out_lens: torch.Tensor,
    ):
        embedding_layer = self.model.llm.get_input_embeddings()
        prompt_embeddings: List[torch.Tensor] = []

        for index, sample in enumerate(samples):
            prepared = sample["prepared"]
            source_ids = prepared["source_ids"][0].to(self.device)
            embeddings = embedding_layer(source_ids).clone()
            fbank_beg = int(prepared["fbank_beg"].reshape(-1)[0].item())
            fake_token_len = int(prepared["fake_token_len"].reshape(-1)[0].item())
            adaptor_len = int(adaptor_out_lens[index].item())

            if fbank_beg < 0 or fake_token_len <= 0:
                raise RuntimeError(
                    f"Invalid speech placeholder: begin={fbank_beg}, length={fake_token_len}"
                )
            if fbank_beg + fake_token_len > embeddings.shape[0]:
                raise RuntimeError("Speech placeholder exceeds the prompt length")
            if fake_token_len > adaptor_out.shape[1]:
                raise RuntimeError(
                    "Audio adaptor output is shorter than the speech placeholder: "
                    f"placeholder={fake_token_len}, padded_adaptor={adaptor_out.shape[1]}"
                )
            if adaptor_len != fake_token_len:
                LOGGER.debug(
                    "Adaptor length differs from placeholder for sample %d: %d vs %d",
                    index,
                    adaptor_len,
                    fake_token_len,
                )

            embeddings[fbank_beg : fbank_beg + fake_token_len] = adaptor_out[
                index, :fake_token_len
            ].to(embeddings.dtype)
            prompt_embeddings.append(embeddings)

        # Decoder-only generation must be left padded so every sample starts
        # generation immediately after its own prompt.
        batch_size = len(prompt_embeddings)
        max_prompt_len = max(item.shape[0] for item in prompt_embeddings)
        hidden_size = prompt_embeddings[0].shape[-1]
        padded = torch.zeros(
            (batch_size, max_prompt_len, hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_prompt_len), dtype=torch.long, device=self.device
        )
        for index, embeddings in enumerate(prompt_embeddings):
            prompt_len = embeddings.shape[0]
            padded[index, -prompt_len:] = embeddings.to(self.dtype)
            attention_mask[index, -prompt_len:] = 1
        return padded, attention_mask

    def _decode_ctc_batch(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        if self.model.ctc_decoder is None:
            return [{} for _ in range(encoder_out.shape[0])]

        decoder_out, _ = self.model.ctc_decoder(encoder_out, encoder_out_lens)
        ctc_logits = self.model.ctc.log_softmax(decoder_out)
        decoded: List[Dict[str, Any]] = []
        for index in range(encoder_out.shape[0]):
            logits = ctc_logits[index, : int(encoder_out_lens[index].item())]
            token_ids = torch.unique_consecutive(logits.argmax(dim=-1), dim=-1)
            token_ids = token_ids[token_ids != self.model.blank_id].tolist()
            decoded.append(
                {
                    "text": self.model.ctc_tokenizer.decode(token_ids).replace(
                        "<|nospeech|>", ""
                    ),
                    "logits": logits,
                }
            )
        return decoded

    def _add_timestamps(self, result: Dict[str, Any], ctc: Dict[str, Any]) -> None:
        from funasr.models.fun_asr_nano.tools.utils import forced_align

        result["ctc_text"] = ctc["text"]
        for text_key, output_key in (
            ("text", "timestamps"),
            ("ctc_text", "ctc_timestamps"),
        ):
            token_ids = torch.tensor(
                self.model.ctc_tokenizer.encode(result[text_key]), dtype=torch.int64
            )
            timestamps = forced_align(ctc["logits"], token_ids, self.model.blank_id)
            for timestamp in timestamps:
                timestamp["token"] = self.model.ctc_tokenizer.decode([timestamp["token"]])
                timestamp["start_time"] = timestamp["start_time"] * 6 * 10 / 1000
                timestamp["end_time"] = timestamp["end_time"] * 6 * 10 / 1000
            result[output_key] = timestamps

    @torch.inference_mode()
    def transcribe_batch(
        self,
        inputs: Sequence[AudioInput],
        keys: Optional[Sequence[str]] = None,
        language: Optional[str] = None,
        hotwords: Sequence[str] = (),
        itn: bool = True,
        return_timestamps: bool = True,
        max_new_tokens: Optional[int] = None,
        llm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Transcribe one physical GPU batch and preserve input order."""
        if not inputs:
            return []
        if keys is None:
            keys = [f"sample_{index:06d}" for index in range(len(inputs))]
        if len(keys) != len(inputs):
            raise ValueError("keys and inputs must have the same length")

        started = time.perf_counter()
        samples = [
            self._prepare_one(audio, language, hotwords, itn) for audio in inputs
        ]
        prepared_at = time.perf_counter()

        encoder_out, encoder_out_lens, adaptor_out, adaptor_out_lens = (
            self._encode_audio_batch(samples)
        )
        prompt_embeds, attention_mask = self._build_prompt_batch(
            samples, adaptor_out, adaptor_out_lens
        )
        encoded_at = time.perf_counter()

        generation_args: Dict[str, Any] = {
            "do_sample": False,
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "pad_token_id": self.pad_token_id,
        }
        if llm_kwargs:
            generation_args.update(llm_kwargs)

        generated_ids = self.model.llm.generate(
            inputs_embeds=prompt_embeds,
            attention_mask=attention_mask,
            **generation_args,
        )
        responses = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        generated_at = time.perf_counter()

        ctc_results = (
            self._decode_ctc_batch(encoder_out, encoder_out_lens)
            if return_timestamps
            else [{} for _ in inputs]
        )
        results: List[Dict[str, Any]] = []
        for key, response, ctc in zip(keys, responses, ctc_results):
            text = re.sub(r"\s+", " ", response.replace("/sil", " ")).strip()
            result: Dict[str, Any] = {
                "key": key,
                "text": text,
                "text_tn": re.sub(r"[^\w\s\u3000\u4e00-\u9fff]+", "", response),
            }
            if return_timestamps and ctc:
                self._add_timestamps(result, ctc)
            results.append(result)
        finished = time.perf_counter()

        self.last_stats = {
            "batch_size": float(len(inputs)),
            "prepare_seconds": prepared_at - started,
            "encode_seconds": encoded_at - prepared_at,
            "generate_seconds": generated_at - encoded_at,
            "ctc_seconds": finished - generated_at,
            "total_seconds": finished - started,
            "max_prompt_tokens": float(prompt_embeds.shape[1]),
        }
        return results

    def generate(
        self,
        inputs: Sequence[AudioInput],
        batch_size: int = 4,
        keys: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Transcribe any number of inputs in fixed-size physical batches."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if keys is None:
            keys = [f"sample_{index:06d}" for index in range(len(inputs))]
        if len(keys) != len(inputs):
            raise ValueError("keys and inputs must have the same length")

        results: List[Dict[str, Any]] = []
        for begin in range(0, len(inputs), batch_size):
            end = min(begin + batch_size, len(inputs))
            batch_results = self.transcribe_batch(
                inputs[begin:end], keys=keys[begin:end], **kwargs
            )
            results.extend(batch_results)
            LOGGER.info(
                "Batch %d:%d finished in %.3fs (encode %.3fs, generate %.3fs)",
                begin,
                end,
                self.last_stats["total_seconds"],
                self.last_stats["encode_seconds"],
                self.last_stats["generate_seconds"],
            )
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fun-ASR-Nano manual FP32 batch inference without vLLM"
    )
    parser.add_argument("audio", nargs="*", help="Audio paths")
    parser.add_argument("--input-list", help="Text file containing one audio path per line")
    parser.add_argument("--output-json", help="Write results as a JSON array")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--language", default="auto")
    parser.add_argument("--hotword", action="append", default=[])
    parser.add_argument("--no-itn", action="store_true")
    parser.add_argument("--no-timestamps", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    audio_paths = list(args.audio)
    if args.input_list:
        audio_paths.extend(
            line.strip()
            for line in Path(args.input_list).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not audio_paths:
        raise SystemExit("No audio input was provided")

    language = LANGUAGE_ALIASES.get(args.language, args.language)
    engine = FunASRNanoBatch(
        model_dir=args.model_dir,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )
    results = engine.generate(
        audio_paths,
        batch_size=args.batch_size,
        keys=[Path(path).stem for path in audio_paths],
        language=language,
        hotwords=args.hotword,
        itn=not args.no_itn,
        return_timestamps=not args.no_timestamps,
    )

    rendered = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        LOGGER.info("Wrote %d results to %s", len(results), output_path)
    else:
        print(rendered)


if __name__ == "__main__":
    main()
