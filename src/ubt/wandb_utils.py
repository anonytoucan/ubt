"""wandb logging for UBT training runs.

- init_wandb: stamps the repo git SHA, liblouis version, data manifest tag+sha and run config into the run.
- GpuStatsCallback: logs nvidia-smi power/temp/util per GPU on every logging step, so throttling is visible.
- ProbeCallback: at every save, greedy-decodes fixed dev docs and logs probe metrics.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess

from transformers import TrainerCallback

_TAG_RE = re.compile(r"^⟨([^⟨⟩]+)⟩(.*)$")


def _manifest_fingerprint(manifest_path: str) -> dict:
    with open(manifest_path, "rb") as fh:
        raw = fh.read()
    data = json.loads(raw)
    return {"manifest_tag": data.get("tag"),
            "manifest_sha256": hashlib.sha256(raw).hexdigest()[:16],
            "manifest_repo_sha": data.get("repo_sha", "")[:8]}


def init_wandb(run_name: str, config: dict, manifest_path: str):
    """Start or resume the wandb run; the id derives from run_name, so a job relaunched with --resume continues the
    same run."""
    import wandb  # noqa: PLC0415

    from ubt.report import louis_version, repo_git_sha  # noqa: PLC0415

    full_config = {
        **config,
        "repo_git_sha": repo_git_sha(),
        "liblouis_version": louis_version(),
        **_manifest_fingerprint(manifest_path),
    }
    run_id = "ubt-" + hashlib.sha1(run_name.encode()).hexdigest()[:12]
    return wandb.init(project=None, name=run_name, id=run_id,
                      resume="allow", config=full_config)


class GpuStatsCallback(TrainerCallback):
    """power.draw/temperature/utilization per GPU into wandb on log steps."""

    def on_log(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,power.draw,temperature.gpu,"
                 "utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout
        except Exception:
            return
        import wandb  # noqa: PLC0415

        stats = {}
        for line in out.strip().splitlines():
            try:
                idx, power, temp, util, mem = [x.strip() for x in line.split(",")]
                stats[f"gpu/{idx}/power_w"] = float(power)
                stats[f"gpu/{idx}/temp_c"] = float(temp)
                stats[f"gpu/{idx}/util_pct"] = float(util)
                stats[f"gpu/{idx}/mem_mb"] = float(mem)
            except (ValueError, IndexError):
                continue  # '[N/A]' / 'ERR!' fields on flaky GPUs
        if stats:
            wandb.log(stats)  # no explicit step: avoids post-final-step clashes


def parse_target(completion: str) -> list[tuple[str, str]]:
    """Parse '⟨table⟩text' lines; returns [] on any malformed line."""
    out = []
    for line in completion.split("\n"):
        m = _TAG_RE.match(line)
        if not m:
            return []
        out.append((m.group(1), m.group(2)))
    return out


class ProbeCallback(TrainerCallback):
    """Greedy-decode fixed dev and train samples; log transcripts and, on dev, probe/parse_rate, probe/table_id_valid
    (every emitted table id in the inventory) and probe/cer."""

    BATCH = 8

    def __init__(self, tokenizer, dev_records: list[dict],
                 train_records: list[dict], hinted: bool,
                 valid_table_ids: set[str],
                 max_new_tokens: int = 768, every_n_saves: int = 1,
                 budget_sec: float = 900.0):
        self.tokenizer = tokenizer
        self.dev_records = dev_records
        self.train_records = train_records
        self.hinted = hinted
        self.valid_ids = valid_table_ids
        self.max_new_tokens = max_new_tokens
        self.every_n_saves = every_n_saves
        self.budget_sec = budget_sec

    def _decode_batch(self, model, records: list[dict],
                      deadline: float) -> tuple[list[dict], bool]:
        """Batched greedy generation, returns (rows, partial). The wall-clock budget keeps rank 0 from stalling the
        other DDP ranks past the NCCL watchdog; a partial probe is logged instead."""
        import time  # noqa: PLC0415

        import torch  # noqa: PLC0415
        from rapidfuzz.distance import Levenshtein  # noqa: PLC0415

        from ubt.task_format import render  # noqa: PLC0415

        enc_dec = bool(getattr(model.config, "is_encoder_decoder", False))
        rows, partial = [], False
        prev_side = self.tokenizer.padding_side
        # Left padding is required for decoder-only batched generation.
        self.tokenizer.padding_side = "right" if enc_dec else "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            for i in range(0, len(records), self.BATCH):
                if time.monotonic() > deadline:
                    partial = True
                    break
                chunk = records[i : i + self.BATCH]
                exs = [render(r, hinted=self.hinted) for r in chunk]
                # add_special_tokens=True matches the training-side prompt encoding
                batch = self.tokenizer([e["prompt"] for e in exs],
                                       return_tensors="pt", padding=True,
                                       add_special_tokens=True).to(model.device)
                with torch.no_grad():
                    gen = model.generate(**batch,
                                         max_new_tokens=self.max_new_tokens,
                                         do_sample=False, use_cache=True,
                                         pad_token_id=self.tokenizer.pad_token_id)
                for j, (rec, ex) in enumerate(zip(chunk, exs)):
                    if enc_dec:
                        out_ids = gen[j]  # seq2seq output IS the completion
                    else:
                        out_ids = gen[j][batch["input_ids"].shape[1]:]
                    hyp = self.tokenizer.decode(out_ids, skip_special_tokens=True)
                    segs = parse_target(hyp)
                    gold = parse_target(ex["completion"])
                    ref = "\n".join(t for _, t in gold)
                    if segs:
                        hyp_text = "\n".join(t for _, t in segs)
                        cer = Levenshtein.distance(ref, hyp_text) / max(1, len(ref))
                    else:
                        # unparsed counts as fully wrong: no survivor bias
                        cer = 1.0
                    from ubt.lang_categories import doc_category  # noqa: PLC0415
                    rows.append({
                        "id": rec["id"], "doc_type": rec["doc_type"],
                        "k": rec["k"],
                        "category": doc_category([t for t, _ in gold]),
                        "gold_tables": ",".join(t for t, _ in gold),
                        "pred_tables": (",".join(t for t, _ in segs)
                                        if segs else "(unparsed)"),
                        "parse_ok": bool(segs),
                        "ids_valid": bool(segs) and all(
                            t in self.valid_ids for t, _ in segs),
                        "ref_text": ref[:300],
                        "hyp_text": ("\n".join(t for _, t in segs)
                                     if segs else hyp)[:300],
                        "cer": round(cer, 4),
                    })
        finally:
            self.tokenizer.padding_side = prev_side
        return rows, partial

    def _run_probe(self, state, model, tag: str) -> None:
        import time  # noqa: PLC0415

        import torch  # noqa: PLC0415
        import wandb  # noqa: PLC0415

        torch.cuda.empty_cache()
        model.eval()
        deadline = time.monotonic() + self.budget_sec
        try:
            dev_rows, p1 = self._decode_batch(model, self.dev_records, deadline)
            train_rows, p2 = self._decode_batch(model, self.train_records, deadline)
            partial = p1 or p2
        except Exception as e:  # noqa: BLE001
            model.train()
            torch.cuda.empty_cache()
            wandb.log({"probe/error": str(e)[:200],
                       "probe/at_step": state.global_step})
            return
        model.train()
        torch.cuda.empty_cache()
        if not dev_rows:
            wandb.log({"probe/error": "budget exhausted before any batch",
                       "probe/at_step": state.global_step})
            return

        n = len(dev_rows)
        parsed = [r for r in dev_rows if r["parse_ok"]]
        valid = [r for r in dev_rows if r["ids_valid"]]
        cers_all = [r["cer"] for r in dev_rows]
        cers_parsed = [r["cer"] for r in parsed]
        cols = ["id", "doc_type", "k", "category", "gold_tables",
                "pred_tables", "parse_ok", "ids_valid", "cer", "ref_text",
                "hyp_text"]
        # per-category / per-language CER, so a collapse in one category is not hidden in the aggregate
        from collections import defaultdict  # noqa: PLC0415

        from ubt.lang_categories import lang_of_table  # noqa: PLC0415
        by_cat, by_lang = defaultdict(list), defaultdict(list)
        for r in dev_rows:
            by_cat[r.get("category", "Latin")].append(r["cer"])
            for t in r["gold_tables"].split(","):
                if t:
                    by_lang[lang_of_table(t)].append(r["cer"])
        lang_metrics = {}
        for c, v in by_cat.items():
            lang_metrics[f"probe_cat/{c}/cer"] = sum(v) / len(v)
            lang_metrics[f"probe_cat/{c}/n"] = len(v)
        for l, v in by_lang.items():
            if len(v) >= 3:
                lang_metrics[f"probe_lang/{l}/cer"] = sum(v) / len(v)
        wandb.log(lang_metrics)
        wandb.log({
            "probe/parse_rate": len(parsed) / n,
            "probe/table_id_valid": len(valid) / n,
            "probe/cer": sum(cers_all) / n,           # unparsed = 1.0, unbiased
            "probe/cer_parsed": (sum(cers_parsed) / len(cers_parsed)
                                 if cers_parsed else 1.0),
            "probe/n": n, "probe/partial": int(partial),
            "probe/at_step": state.global_step,
            f"transcripts/{tag}/dev": wandb.Table(
                columns=cols, data=[[r[c] for c in cols] for r in dev_rows]),
            f"transcripts/{tag}/train": wandb.Table(
                columns=cols, data=[[r[c] for c in cols] for r in train_rows]),
        })

    def on_save(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero or model is None:
            return
        # cadence from global_step, so a resumed run keeps the same probe schedule
        interval = max(1, int(args.save_steps)) * self.every_n_saves
        if state.global_step % interval == 0:
            self._run_probe(state, model, f"step{state.global_step}")

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero or model is None:
            return
        self._run_probe(state, model, f"epoch{round(state.epoch or 0)}")
