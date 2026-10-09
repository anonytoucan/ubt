"""Paper metrics beyond CER/bCER: embedding similarity (sim.), LLM judge (judge), BLEU / chrF++.

  sim.  = per-document cosine of BAAI/bge-m3 dense embeddings (CLS pooling, L2-normalised) of hypothesis and
          reference, reported as the mean x 100; bge-m3 is multilingual with an 8k context.
  judge = Qwen2.5-72B-Instruct-AWQ (local vLLM) rates fidelity to the reference on 1-5; the score is the expected
          digit under the first output position's probabilities, reported as the mean. An empty output scores 1
          without asking, since the judge rates empty outputs highly.
  BLEU / chrF++ = sacrebleu corpus scores (chrF++ = chrF with word_order=2).
Texts are the joined segment texts ("\\n".join), as CER uses them.
"""

from __future__ import annotations

import math

SIM_MODEL = "BAAI/bge-m3"
JUDGE_MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"

JUDGE_SYSTEM = (
    "You evaluate the output of a braille-to-text transcription system. The REFERENCE is the exact "
    "original text; the OUTPUT is the system's reading of its braille. Judge fidelity to the reference: "
    "meaning, wording, names, numbers and symbols. Ignore differences that are only whitespace.\n"
    "5 = identical, or differs only in insignificant formatting\n"
    "4 = minor slips that do not change the meaning (e.g. one wrong character in a common word)\n"
    "3 = errors that change details (a name, number, date, or content word) but keep the main meaning\n"
    "2 = major errors; the meaning is only partly recoverable\n"
    "1 = mostly wrong, garbled, or unrelated\n"
    "Answer with a single digit (1-5) and nothing else.")


def judge_prompt(tokenizer, reference: str, output: str, lang: str | None = None) -> str:
    user = (f"Language/code: {lang}\n" if lang else "") + \
        f"REFERENCE:\n{reference}\n\nOUTPUT:\n{output}\n\nScore (1-5):"
    msgs = [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def expected_digit(logprobs: dict | None) -> float | None:
    """vLLM top-k logprobs at the first output position -> E[score] over '1'..'5' (renormalised)."""
    if not logprobs:
        return None
    mass = {}
    for tok_id, lp in logprobs.items():
        t = (getattr(lp, "decoded_token", None) or "").strip()
        if t in {"1", "2", "3", "4", "5"}:
            mass[int(t)] = mass.get(int(t), 0.0) + math.exp(lp.logprob)
    z = sum(mass.values())
    if z <= 0:
        return None
    return sum(k * v for k, v in mass.items()) / z


def run_judge(pairs: list[tuple[str, str, str | None]], model: str = JUDGE_MODEL, tp: int = 2,
              max_model_len: int = 8192, gpu_mem: float = 0.90) -> list[float | None]:
    """pairs: [(reference, output, lang)] -> expected 1-5 scores."""
    from vllm import LLM, SamplingParams  # noqa: PLC0415
    llm = LLM(model=model, tensor_parallel_size=tp, max_model_len=max_model_len,
              gpu_memory_utilization=gpu_mem, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    ask = [i for i, (_, out, _) in enumerate(pairs) if out.strip()]      # an output with no text is scored 1
    prompts = [judge_prompt(tok, pairs[i][0][:6000], pairs[i][1][:6000], pairs[i][2]) for i in ask]
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    outs = llm.generate(prompts, sp) if prompts else []
    res: list[float | None] = [1.0] * len(pairs)
    for i, o in zip(ask, outs):
        res[i] = expected_digit(o.outputs[0].logprobs[0] if o.outputs and o.outputs[0].logprobs else None)
    return res


def run_sim(refs: list[str], hyps: list[str], model: str = SIM_MODEL, batch: int = 64,
            device: str = "cuda") -> list[float]:
    """Per-pair cosine of bge-m3 dense embeddings (CLS pooling, normalised)."""
    import torch  # noqa: PLC0415
    from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415
    tok = AutoTokenizer.from_pretrained(model)
    enc = AutoModel.from_pretrained(model, torch_dtype=torch.float16).to(device).eval()

    def embed(texts):
        out = []
        for i in range(0, len(texts), batch):
            b = tok(texts[i:i + batch], padding=True, truncation=True, max_length=2048,
                    return_tensors="pt").to(device)
            with torch.no_grad():
                h = enc(**b).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu())
        return torch.cat(out)
    er, eh = embed(refs), embed([h if h else " " for h in hyps])
    return (er * eh).sum(-1).tolist()


def bleu_chrf(refs: list[str], hyps: list[str], tokenize: str = "13a") -> dict:
    """Corpus BLEU and chrF++ (sacrebleu). For Chinese use tokenize='zh'."""
    import sacrebleu  # noqa: PLC0415
    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tokenize)
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)
    return {"bleu": round(bleu.score, 2), "chrf++": round(chrf.score, 2)}
