import os
import json
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity


class SegmentRetriever:
    def __init__(self, model_name, seg_dir, orig_dir, output_dir, top_k=10):
        self.seg_dir, self.orig_dir, self.output_dir = Path(seg_dir), Path(orig_dir), Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
        self.model.eval()
        self.top_k = top_k
        self.hidden_size = self.model.config.hidden_size

    def get_mean_embedding(self, texts):
        embs = []
        for i in range(0, len(texts), 4):
            inputs = self.tokenizer(
                texts[i:i+4],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=32768
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            mean = (out * mask).sum(1) / mask.sum(1)
            # ✅ bfloat16 → float32 변환 추가
            embs.append(mean.to(torch.float32).cpu().numpy())
        return np.vstack(embs)

    def compute_segment_embeddings(self, segments):
        seg_embs, valid_idx = [], []
        for idx, seg in enumerate(segments):
            sents = seg.get("sentences", [])
            if not sents: continue
            emb = self.get_mean_embedding(sents)
            seg_embs.append(np.mean(emb, axis=0, keepdims=True))
            valid_idx.append(idx)
        return valid_idx, np.vstack(seg_embs)

    def run(self):
        for seg_file in sorted(self.seg_dir.glob("*_segments.json")):
            orig_file = self.orig_dir / seg_file.name.replace("_segments.json", ".json")
            if not orig_file.exists(): continue
            with open(seg_file, "r", encoding="utf-8") as f:
                segments = json.load(f)["segments"]
            with open(orig_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            subtopics = [t["topic"] for t in data.get("topic_summary", [])]
            valid_idx, seg_embs = self.compute_segment_embeddings(segments)
            sub_embs = self.get_mean_embedding(subtopics)
            sims = cosine_similarity(sub_embs, seg_embs)
            topic_segments = {}
            for s_idx, topic in enumerate(subtopics):
                sim_row = sims[s_idx]
                top_idx = np.argsort(-sim_row)[:min(self.top_k, len(sim_row))]
                topic_segments[topic] = [
                    {"segment_id": valid_idx[i],
                     "avg_similarity": float(sim_row[i]),
                     "sentences": segments[valid_idx[i]]["sentences"],
                     "rank": r+1} for r, i in enumerate(top_idx)
                ]
            out_path = self.output_dir / seg_file.name.replace("_segments.json", "_topic_segments.json")
            json.dump(topic_segments, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            print(f"💾 {out_path.name} 저장 완료")
