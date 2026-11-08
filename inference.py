import json
import os
import re
 
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
 
ESC_RE = re.compile(r"яя\s*(\d+)\s*яя")
 
 
class SmartHomeModel:
    def __init__(self, path_or_repo, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(path_or_repo)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path_or_repo).to(self.device).eval()
        if os.path.isdir(path_or_repo):
            esc_path = os.path.join(path_or_repo, "escape_chars.json")
        else:
            esc_path = hf_hub_download(path_or_repo, "escape_chars.json")
        with open(esc_path, encoding="utf-8") as f:
            self.bad = set(json.load(f))
 
    def _enc(self, s):
        return "".join(f"яя{ord(c)}яя" if c in self.bad else c for c in s)
 
    def _dec(self, s):
        return ESC_RE.sub(lambda m: chr(int(m.group(1))), s).strip()
 
    @torch.no_grad()
    def predict(self, queries, max_new_tokens=160):
        single = isinstance(queries, str)
        if single:
            queries = [queries]
        texts = [self._enc(f"query: {q}") for q in queries]
        enc = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=64
        ).to(self.device)
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)
        decoded = [self._dec(t) for t in self.tokenizer.batch_decode(out, skip_special_tokens=True)]
        results = []
        for t in decoded:
            try:
                calls = json.loads(t)
            except Exception:
                calls = None
            results.append({"text": t, "calls": calls})
        return results[0] if single else results
 
 
if __name__ == "__main__":
    import sys
 
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    model = SmartHomeModel(path)
    print(json.dumps(model.predict("включи свет на кухне"), ensure_ascii=False, indent=2))
