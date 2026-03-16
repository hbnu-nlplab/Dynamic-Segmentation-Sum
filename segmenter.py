import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from pathlib import Path


def setup_korean_font():
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rc("font", family="NanumGothic")
    else:
        plt.rc("font", family="DejaVu Sans")
    plt.rc("axes", unicode_minus=False)


class MeetingSegmenter:
    def __init__(self, model_name, input_dir, output_dir):
        self.model_name = model_name
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.vis_dir = self.output_dir / "visualizations"
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
        self.model.eval()

        if self.tokenizer.sep_token is None or "<sep>" not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"sep_token": "<sep>"})
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.sep_token = self.tokenizer.sep_token
        self.sep_token_id = self.tokenizer.sep_token_id

    def get_embeddings(self, sentences, max_length=32768):
        sep_token = self.sep_token
        all_embeddings = []
        total_sent = len(sentences)
        start_idx = 0

        while start_idx < total_sent:
            joined_text = f" {sep_token} ".join(sentences[start_idx:]) + f" {sep_token}"
            enc = self.tokenizer(
                joined_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                add_special_tokens=True
            ).to(self.model.device)

            input_ids = enc["input_ids"][0].cpu().numpy()
            sep_positions = np.where(input_ids == self.sep_token_id)[0]
            covered_sentences = len(sep_positions)
            if covered_sentences == 0:
                break

            with torch.no_grad():
                outputs = self.model(**enc)
                hidden_states = outputs.last_hidden_state.squeeze(0)
                embs = hidden_states[sep_positions].to(torch.float32).cpu().numpy()
            all_embeddings.extend(embs)

            start_idx += covered_sentences
        return np.array(all_embeddings)

    def segment_sentences(self, sentences, embeddings):
        similarities = [
            cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            for i in range(len(embeddings) - 1)
        ]
        similarities = np.array(similarities)
        z_thresh = np.mean(similarities) - 1.5 * np.std(similarities)
        p_thresh = np.percentile(similarities, 5)
        threshold = (z_thresh + p_thresh) / 2
        change_points = np.where(similarities < threshold)[0] + 1

        segments, start = [], 0
        for cp in change_points:
            segments.append(sentences[start:cp])
            start = cp
        segments.append(sentences[start:])
        return segments, similarities, threshold, change_points

    def segment_embedding(self, embs):
        return np.mean(embs, axis=0, keepdims=True)

    def merge_small_segments(self, segments, embeddings, min_len=5):
        seg_ranges = []
        idx = 0
        for seg in segments:
            seg_ranges.append((idx, idx + len(seg)))
            idx += len(seg)

        merged = [seg.copy() for seg in segments]
        ranges = seg_ranges.copy()

        i = 0
        while i < len(merged):
            if len(merged[i]) > min_len:
                i += 1
                continue

            cur_s, cur_e = ranges[i]
            cur_emb = self.segment_embedding(embeddings[cur_s:cur_e])

            sim_prev, sim_next = -1, -1

            if i > 0:
                ps, pe = ranges[i - 1]
                prev_emb = self.segment_embedding(embeddings[ps:pe])
                sim_prev = cosine_similarity(cur_emb, prev_emb)[0][0]

            if i < len(merged) - 1:
                ns, ne = ranges[i + 1]
                next_emb = self.segment_embedding(embeddings[ns:ne])
                sim_next = cosine_similarity(cur_emb, next_emb)[0][0]

            if sim_prev >= sim_next and i > 0:
                merged[i - 1].extend(merged[i])
                ranges[i - 1] = (ranges[i - 1][0], ranges[i][1])
                del merged[i]
                del ranges[i]
                i -= 1
            elif i < len(merged) - 1:
                merged[i].extend(merged[i + 1])
                ranges[i] = (ranges[i][0], ranges[i + 1][1])
                del merged[i + 1]
                del ranges[i + 1]
            else:
                i += 1

        return merged

    def visualize(self, embeddings, similarities, threshold, change_points, save_prefix):
        sim_matrix = cosine_similarity(embeddings)
        plt.figure(figsize=(8, 8))
        im = plt.imshow(sim_matrix, cmap="hot_r", interpolation="nearest")
        plt.colorbar(im)
        plt.title("문장-문장 유사도 (붉은색)")
        for cp in change_points:
            plt.axhline(cp - 0.5, color="lime", linestyle="--")
            plt.axvline(cp - 0.5, color="lime", linestyle="--")
        plt.tight_layout()
        plt.savefig(save_prefix + "_heatmap.png")
        plt.close()

    def run(self):
        setup_korean_font()
        for file in tqdm(sorted(self.input_dir.glob("*.json"))):
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            sentences = [u["sentence"] for u in data["dialogue"]]
            embeddings = self.get_embeddings(sentences)

            segments, sims, th, cps = self.segment_sentences(sentences, embeddings)

            merged = self.merge_small_segments(segments, embeddings, min_len=5)

            merged_cp, idx = [], 0
            for seg in merged[:-1]:
                idx += len(seg)
                merged_cp.append(idx)

            out_path = self.output_dir / f"{file.stem}_segments.json"
            json.dump(
                {"segments": [{"id": i, "sentences": seg} for i, seg in enumerate(merged)]},
                open(out_path, "w", encoding="utf-8"),
                ensure_ascii=False,
                indent=2
            )

            vis_prefix = str(self.vis_dir / f"{file.stem}_analysis")
            self.visualize(embeddings, sims, th, merged_cp, vis_prefix)

            print(f"✅ {file.name} 처리 완료")