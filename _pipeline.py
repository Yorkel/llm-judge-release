"""Judge call pipeline for paradigm 2 (LLM-as-judge).

Implements:
- Two-round bidirectional position swap (Arena-Hard repo default)
- Prompt caching via prompt_cache_key (per-cell stable prefix)
- Verdict extraction with primary + fallback regex (matches v2.0 config)
- Resumable batch runner (skip already-judged uids)
"""

import concurrent.futures
import json
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from _prompts import get_system_prompt, get_user_prompt

# v2.0 default judge config (from arena-hard-v2.0.yaml)
DEFAULT_JUDGE_MODEL = "gpt-4.1"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 16000

# Verdict extraction regex (v2.0 default)
PRIMARY_REGEX = re.compile(r"\[\[([AB<>=]+)\]\]")
FALLBACK_REGEX = re.compile(r"\[([AB<>=]+)\]")

# Alternatives kept per verdict-tag token when logprobs are requested (API max 20)
TOP_LOGPROBS = 20


def parse_verdict(text: str) -> Optional[str]:
    """Extract verdict tag from judge output. Tries primary then fallback regex."""
    upper = text.upper()
    matches = [m for m in PRIMARY_REGEX.findall(upper) if m]
    if matches:
        return matches[-1].strip("\n")
    matches = [m for m in FALLBACK_REGEX.findall(upper) if m]
    if matches:
        return matches[-1].strip("\n")
    return None


def verdict_token_logprobs(content) -> Optional[list]:
    """Logprobs for the tokens of the verdict tag that parse_verdict reads.

    content is resp.choices[0].logprobs.content. Finds the last [[...]] tag (or
    the [...] fallback) in the token text and keeps only the tokens overlapping
    it, each with its top alternatives. Matches case-insensitively on the
    original text, so character positions line up with the tokens.
    """
    if not content:
        return None
    text = "".join(t.token for t in content)
    primary = re.compile(PRIMARY_REGEX.pattern, re.IGNORECASE)
    fallback = re.compile(FALLBACK_REGEX.pattern, re.IGNORECASE)
    matches = list(primary.finditer(text)) or list(fallback.finditer(text))
    if not matches:
        return None
    start, end = matches[-1].span()
    kept, pos = [], 0
    for t in content:
        t_start, pos = pos, pos + len(t.token)
        if t_start < end and pos > start:
            kept.append({
                "token": t.token,
                "logprob": t.logprob,
                "top": [{"token": a.token, "logprob": a.logprob} for a in (t.top_logprobs or [])],
            })
    return kept


def call_judge(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    cell_id: str,
    model: str = DEFAULT_JUDGE_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_retries: int = 8,
    logprobs: bool = False,
) -> dict:
    """Single judge call with caching + retry-on-429.

    Retries with exponential backoff on rate-limit errors (429). The OpenAI
    SDK's default max_retries=2 isn't enough when many workers hit TPM cap
    simultaneously.

    logprobs=True requests token logprobs and keeps only those for the verdict
    tag. Off by default, so the May runners send the same request as before.
    """
    logprob_args = {"logprobs": True, "top_logprobs": TOP_LOGPROBS} if logprobs else {}
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"prompt_cache_key": f"judge_{cell_id}"},
                **logprob_args,
            )
            break  # success
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower() or "tpm" in err_str.lower()
            if is_rate_limit and attempt < max_retries - 1:
                # Exponential backoff: 2, 4, 8, 16, 32, 64, 128 seconds
                wait = 2 ** (attempt + 1)
                time.sleep(wait)
                continue
            raise  # non-retryable error or final retry exhausted
    raw = resp.choices[0].message.content
    verdict = parse_verdict(raw)
    usage = resp.usage
    cached_tokens = (
        getattr(usage.prompt_tokens_details, "cached_tokens", 0)
        if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details
        else 0
    )
    record = {
        "raw": raw,
        "verdict": verdict,
        "parse_success": verdict is not None,
        "input_tokens": usage.prompt_tokens,
        "cached_tokens": cached_tokens,
        "output_tokens": usage.completion_tokens,
        "system_fingerprint": getattr(resp, "system_fingerprint", None),
        "served_model": resp.model,
    }
    if logprobs:
        lp = resp.choices[0].logprobs
        record["verdict_logprobs"] = verdict_token_logprobs(lp.content if lp else None)
    return record


def run_judge_pair(
    client: OpenAI,
    cell_id: str,
    question: str,
    candidate_response: str,
    baseline_response: str,
    **kwargs,
) -> dict:
    """Two-round bidirectional swap for one (question, candidate, baseline) triple.

    Round 1: A=baseline, B=candidate (matches Arena-Hard convention).
    Round 2: A=candidate, B=baseline (positions swapped).
    """
    system_prompt = get_system_prompt(cell_id)
    user_r1 = get_user_prompt(cell_id, question, answer_a=baseline_response, answer_b=candidate_response)
    r1 = call_judge(client, system_prompt, user_r1, cell_id, **kwargs)
    user_r2 = get_user_prompt(cell_id, question, answer_a=candidate_response, answer_b=baseline_response)
    r2 = call_judge(client, system_prompt, user_r2, cell_id, **kwargs)
    return {"round1": r1, "round2": r2}


def load_done_uids(output_path: Path) -> set:
    """uids already written to output_path, for resume.

    A crash mid-write can leave a last line with no newline. It is cut off here,
    so the next append starts on a clean line and that uid is judged again.
    """
    done_uids = set()
    if not output_path.exists():
        return done_uids
    with open(output_path, "rb+") as f:
        data = f.read()
        end = len(data)
        if data and not data.endswith(b"\n"):
            end = data.rfind(b"\n") + 1
            f.truncate(end)
            print(f"[{output_path.name}] removed a partial last line left by an interrupted write")
    for line in data[:end].splitlines():
        if line.strip():
            done_uids.add(json.loads(line)["uid"])
    return done_uids


def run_cell_batch(
    client: OpenAI,
    cell_id: str,
    pairs: pd.DataFrame,
    output_path: Path,
    resume: bool = True,
    **kwargs,
) -> None:
    """Run all (uid, prompt, candidate, baseline) rows for one OAT cell.

    Writes JSONL incrementally for resumability. pairs schema:
      uid, prompt, candidate_response, baseline_response
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_uids = load_done_uids(output_path) if resume else set()
    if done_uids:
        print(f"[{cell_id}] resuming — {len(done_uids)} uids already done")

    to_run = pairs[~pairs["uid"].isin(done_uids)]
    if len(to_run) == 0:
        print(f"[{cell_id}] already complete")
        return

    with open(output_path, "a", encoding="utf-8") as f:
        for _, row in tqdm(to_run.iterrows(), total=len(to_run), desc=cell_id):
            try:
                result = run_judge_pair(
                    client,
                    cell_id,
                    question=row["prompt"],
                    candidate_response=row["candidate_response"],
                    baseline_response=row["baseline_response"],
                    **kwargs,
                )
                record = {
                    "uid": row["uid"],
                    "spec_cell_id": cell_id,
                    "round1": result["round1"],
                    "round2": result["round2"],
                    "model": kwargs.get("model", DEFAULT_JUDGE_MODEL),
                    "timestamp": time.time(),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
            except Exception as e:
                print(f"[{cell_id}] error on uid={row['uid']}: {e}")
                continue


def run_cell_batch_parallel(
    client: OpenAI,
    cell_id: str,
    pairs: pd.DataFrame,
    output_path: Path,
    resume: bool = True,
    max_workers: int = 16,
    **kwargs,
) -> None:
    """Parallel version of run_cell_batch.

    Each worker handles one (uid, cell_id) — both rounds run sequentially per uid
    so output is atomic per row. Multiple uids processed concurrently across
    max_workers threads.

    For full factorial (8,000 calls), max_workers=16 brings runtime from ~47h
    sequential down to ~3h. max_workers=32 → ~1.5h. Match Arena-Hard's gpt-4.1
    config (parallel: 64) if rate-limit headroom allows.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_uids = load_done_uids(output_path) if resume else set()
    if done_uids:
        print(f"[{cell_id}] resuming — {len(done_uids)} uids already done")

    to_run = pairs[~pairs["uid"].isin(done_uids)]
    if len(to_run) == 0:
        print(f"[{cell_id}] already complete")
        return

    rows = to_run.to_dict("records")

    def process_one(row):
        try:
            result = run_judge_pair(
                client,
                cell_id,
                question=row["prompt"],
                candidate_response=row["candidate_response"],
                baseline_response=row["baseline_response"],
                **kwargs,
            )
            return {
                "uid": row["uid"],
                "spec_cell_id": cell_id,
                "round1": result["round1"],
                "round2": result["round2"],
                "model": kwargs.get("model", DEFAULT_JUDGE_MODEL),
                "timestamp": time.time(),
            }
        except Exception as e:
            print(f"[{cell_id}] error on uid={row['uid']}: {e}")
            return None

    with open(output_path, "a", encoding="utf-8") as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_one, row) for row in rows]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"{cell_id} (parallel x{max_workers})",
            ):
                record = future.result()
                if record is not None:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
