"""
generation.py
-------------
Backend-agnostic answer generation. Produces, for each query+evidence:
  - answer text
  - a token-probability confidence signal (Signal 1)

LOCAL backend (Qwen2.5-1.5B-Instruct):
  Uses HuggingFace generate() with output_scores=True to get REAL per-token
  log probabilities. Signal 1 = mean token probability, exactly as designed.

CLAUDE backend (claude API, no logprobs available):
  Signal 1 is approximated by SELF-CONSISTENCY: sample the answer k times at
  temperature>0 and measure agreement (fraction of samples whose normalized
  answer equals the modal answer). High agreement -> high confidence.
  This is the honest substitute for logprobs on a closed API.

GROQ / OPENROUTER backends (free tiers, OpenAI-compatible APIs):
  Same self-consistency proxy as Claude. Keys are read from the git-ignored
  .env file. Free-tier daily quotas raise DailyLimitReached so a checkpointed
  run stops cleanly and resumes later.

All backends share the SAME prompt template so the comparison is fair.
"""
import os
import time
import numpy as np
from collections import Counter

PROMPT_TEMPLATE = (
    "Answer the question using ONLY the evidence below. "
    "Give a short, direct answer (a few words). If the evidence is insufficient, say 'unknown'.\n\n"
    "Evidence:\n{evidence}\n\n"
    "Question: {question}\nAnswer:"
)


def build_prompt(question, evidence_texts):
    ev = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(evidence_texts))
    return PROMPT_TEMPLATE.format(evidence=ev, question=question)


# ----------------------------------------------------------------------
# LOCAL BACKEND
# ----------------------------------------------------------------------
class LocalGenerator:
    def __init__(self, model_name, device="cuda", max_new_tokens=64, device_map=None, load_4bit=False):
        """device_map="auto" splits the model over every visible GPU (e.g. a 7-8B
        model in fp16 on Kaggle's 2x T4); load_4bit quantizes the weights with
        bitsandbytes so a 7-8B model fits one 16 GB GPU (this changes S1)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        print(f"loading local model {model_name} on {device_map or device}"
              f"{' in 4-bit' if load_4bit else ''} ...")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        extra = {}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            extra["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device_map or device,
            **extra,
        )
        self.model.eval()
        self.device = self.model.device         # inputs go to the first shard

    def generate(self, question, evidence_texts):
        prompt = build_prompt(question, evidence_texts)
        messages = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,                 # deterministic
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tok.eos_token_id,
            )
        # decode only the newly generated tokens
        gen_ids = out.sequences[0][inputs["input_ids"].shape[1]:]
        answer = self.tok.decode(gen_ids, skip_special_tokens=True).strip()

        # --- Signal 1: real per-token probabilities ---
        token_logprobs = []
        for i, score in enumerate(out.scores):           # one score tensor per step
            logprobs = self.torch.log_softmax(score[0], dim=-1)
            tok_id = gen_ids[i]
            token_logprobs.append(float(logprobs[tok_id].item()))
        return {
            "answer": answer,
            "token_logprobs": token_logprobs,    # natural log probs
            "signal1_kind": "token_logprob",
        }

    def sample(self, question, evidence_texts, n, temperature=0.7):
        """n sampled answers in one batched generate() call (for sampling-based
        confidence such as self-consistency or semantic entropy)."""
        prompt = build_prompt(question, evidence_texts)
        text = self.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                            tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True, temperature=temperature,
                num_return_sequences=n,
                pad_token_id=self.tok.eos_token_id,
            )
        start = inputs["input_ids"].shape[1]
        return [self.tok.decode(seq[start:], skip_special_tokens=True).strip() for seq in out]


# ----------------------------------------------------------------------
# CLAUDE BACKEND
# ----------------------------------------------------------------------
class ClaudeGenerator:
    def __init__(self, model_name, n_samples=3, max_tokens=64):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model_name
        self.n_samples = n_samples
        self.max_tokens = max_tokens
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set. Run: "
                               "$env:ANTHROPIC_API_KEY = 'sk-ant-...'")

    @staticmethod
    def _norm(s):
        return " ".join(s.lower().strip().split())

    def generate(self, question, evidence_texts):
        prompt = build_prompt(question, evidence_texts)
        samples = []
        for _ in range(self.n_samples):
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.7,                 # >0 so samples can differ
                messages=[{"role": "user", "content": prompt}],
            )
            txt = "".join(b.text for b in msg.content if b.type == "text").strip()
            samples.append(txt)
        # modal answer + self-consistency confidence
        norms = [self._norm(s) for s in samples]
        counts = Counter(norms)
        modal_norm, modal_count = counts.most_common(1)[0]
        consistency = modal_count / len(samples)         # in [0,1]
        # pick a representative raw answer matching the modal normalized form
        answer = next(s for s in samples if self._norm(s) == modal_norm)
        return {
            "answer": answer,
            "self_consistency": consistency,    # used as Signal 1 proxy
            "samples": samples,
            "signal1_kind": "self_consistency",
        }


def load_dotenv(path=os.path.join(os.path.dirname(__file__), "..", ".env")):
    """Read KEY=VALUE lines from the git-ignored .env into os.environ, so API
    keys never live in code or on the command line. Never overrides a key
    that is already set."""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip() and not os.environ.get(k.strip()):
                os.environ[k.strip()] = v.strip().strip('"').strip("'")


def extra_body_for(table, model):
    """Per-model request extras (e.g. reasoning settings), looked up by the
    longest model-id prefix in `table`; models with no match get none."""
    matches = [p for p in table if model.startswith(p)]
    return table[max(matches, key=len)] if matches else {}


def get_generator(backend, config, device="cuda"):
    load_dotenv()
    if backend in OPENAI_COMPAT_PROVIDERS:
        model = config[f"{backend}_model"]
        return OpenAICompatGenerator(backend, model,
                                     n_samples=config["n_consistency_samples"],
                                     max_tokens=config["api_max_tokens"],
                                     rpm=config[f"{backend}_rpm"],
                                     extra_body=extra_body_for(config[f"{backend}_extra_body"], model))
    if backend == "local":
        return LocalGenerator(config["local_model"], device=device,
                              max_new_tokens=config["max_new_tokens"],
                              device_map=config.get("local_device_map"),
                              load_4bit=config.get("local_4bit", False))
    elif backend == "claude":
        return ClaudeGenerator(config["claude_model"],
                               n_samples=config["n_consistency_samples"],
                               max_tokens=config["max_new_tokens"])
    elif backend == "ollama":
        return OllamaGenerator(config["ollama_model"],
                               n_samples=config["n_consistency_samples"],
                               max_tokens=config["max_new_tokens"])
    raise ValueError(f"unknown backend: {backend}")


# ----------------------------------------------------------------------
# OLLAMA BACKEND (open-source local model, no logprobs -> self-consistency)
# ----------------------------------------------------------------------
class OllamaGenerator:
    def __init__(self, model_name, n_samples=3, max_tokens=64):
        import ollama
        self.ollama = ollama
        self.model = model_name
        self.n_samples = n_samples
        self.max_tokens = max_tokens
        # fail early with a clear message if model isn't pulled
        try:
            ollama.show(model_name)
        except Exception as e:
            raise RuntimeError(
                f"Ollama model '{model_name}' not found. Run:  ollama pull {model_name}\n"
                f"(and make sure the Ollama app/service is running)\nDetail: {str(e)[:120]}")

    @staticmethod
    def _norm(s):
        return " ".join(s.lower().strip().split())

    def generate(self, question, evidence_texts):
        prompt = build_prompt(question, evidence_texts)
        samples = []
        for _ in range(self.n_samples):
            resp = self.ollama.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0.7, "num_predict": self.max_tokens},
            )
            samples.append(resp["response"].strip())
        norms = [self._norm(s) for s in samples]
        counts = Counter(norms)
        modal_norm, modal_count = counts.most_common(1)[0]
        consistency = modal_count / len(samples)
        answer = next(s for s in samples if self._norm(s) == modal_norm)
        return {
            "answer": answer,
            "self_consistency": consistency,
            "samples": samples,
            "signal1_kind": "self_consistency",
        }


# ----------------------------------------------------------------------
# OPENAI-COMPATIBLE BACKENDS (Groq, OpenRouter free tiers; no logprobs on
# most free models -> self-consistency, like Claude/Ollama)
# ----------------------------------------------------------------------
OPENAI_COMPAT_PROVIDERS = {
    # backend name: (base URL, env var holding the key)
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}


class DailyLimitReached(RuntimeError):
    """The provider's per-day quota is spent. Traces are checkpointed per
    question, so the run can simply be resumed once the quota resets."""


class OpenAICompatGenerator:
    def __init__(self, provider, model_name, n_samples=3, max_tokens=512, rpm=None,
                 extra_body=None, max_wait=600):
        from openai import OpenAI
        base_url, key_var = OPENAI_COMPAT_PROVIDERS[provider]
        key = os.environ.get(key_var)
        if not key:
            raise RuntimeError(f"{key_var} not set. Put a line {key_var}=... in the repo's "
                               f".env file (git-ignored); see .env.example.")
        self.client = OpenAI(base_url=base_url, api_key=key, max_retries=0, timeout=120)
        self.provider = provider
        self.model = model_name
        self.n_samples = n_samples
        self.max_tokens = max_tokens          # reasoning models spend these on thinking first
        self.extra_body = extra_body or {}
        self.min_interval = 60.0 / rpm if rpm else 0.0
        self.max_wait = max_wait              # a longer Retry-After means the daily quota is gone
        self._last = 0.0
        self.usage = Counter()                # calls / tokens / empty answers, for quota tracking

    _norm = staticmethod(OllamaGenerator._norm)

    @staticmethod
    def _retry_after(err):
        h = getattr(getattr(err, "response", None), "headers", None) or {}
        if h.get("retry-after"):                       # Groq: seconds
            return float(h["retry-after"])
        if h.get("x-ratelimit-reset"):                 # OpenRouter: epoch milliseconds
            return max(0.0, float(h["x-ratelimit-reset"]) / 1000 - time.time())
        return None

    def _call(self, prompt):
        import openai
        for attempt in range(8):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=[{"role": "user", "content": prompt}],
                    temperature=0.7, max_tokens=self.max_tokens, extra_body=self.extra_body)
            except openai.RateLimitError as e:
                retry = self._retry_after(e)
                msg = str(e).lower()
                if "per day" in msg or "per-day" in msg or (retry or 0) > self.max_wait:
                    raise DailyLimitReached(f"{self.provider} daily quota reached: {str(e)[:200]}")
                time.sleep(retry if retry is not None else min(60, 2 ** attempt))
                continue
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError):
                time.sleep(min(60, 2 ** attempt))
                continue
            u = resp.usage
            self.usage["calls"] += 1
            if u is not None:
                self.usage["prompt_tokens"] += u.prompt_tokens or 0
                self.usage["completion_tokens"] += u.completion_tokens or 0
                details = getattr(u, "prompt_tokens_details", None)
                self.usage["cached_tokens"] += getattr(details, "cached_tokens", 0) or 0
            text = (resp.choices[0].message.content or "").strip() if resp.choices else ""
            self.usage["empty_answers"] += int(not text)
            return text
        raise RuntimeError(f"{self.provider}: giving up after repeated errors")

    def generate(self, question, evidence_texts):
        prompt = build_prompt(question, evidence_texts)
        samples = [self._call(prompt) for _ in range(self.n_samples)]
        norms = [self._norm(s) for s in samples]
        modal_norm, modal_count = Counter(norms).most_common(1)[0]
        answer = next(s for s in samples if self._norm(s) == modal_norm)
        return {
            "answer": answer,
            "self_consistency": modal_count / len(samples),
            "samples": samples,
            "signal1_kind": "self_consistency",
        }
