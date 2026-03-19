# evaluator.py

from konlpy.tag import Komoran
from rouge import Rouge
from pathlib import Path
import json
from tqdm import tqdm


class SummaryEvaluator:
    def __init__(self, pred_dir, gt_dir, output_path):
        self.pred_dir = Path(pred_dir)
        self.gt_dir = Path(gt_dir)
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.komoran = Komoran()
        self.scorer = Rouge()

        self.global_scores = {
            "r1": [],
            "r2": [],
            "rl": []
        }

    def morph(self, text):
        return " ".join(self.komoran.morphs(text))
    
    # def morph(self, text):
        # return text  # 형태소 분석 제거


    def run(self):
        results = []

        for pred_file in tqdm(self.pred_dir.glob("*_total_generated_summary.json")):
            base_name = pred_file.stem.replace("_total_generated_summary", "")
            gt_file = self.gt_dir / f"{base_name}.json"

            if not gt_file.exists():
                continue

            with open(pred_file, encoding="utf-8") as f:
                pred_data = json.load(f)

            with open(gt_file, encoding="utf-8") as f:
                gt_data = json.load(f)

            pred = pred_data.get("summary", "").strip()
            gold = gt_data.get("total_summary", [{}])[0].get("total_asummary", "").strip()

            if not pred or not gold:
                continue

            score = self.scorer.get_scores(
                self.morph(gold),
                self.morph(pred)
            )[0]

            r1 = score["rouge-1"]["r"]
            r2 = score["rouge-2"]["r"]
            rl = score["rouge-l"]["r"]

            self.global_scores["r1"].append(r1)
            self.global_scores["r2"].append(r2)
            self.global_scores["rl"].append(rl)

            results.append({
                "filename": pred_file.name,
                "rouge-1": round(r1, 4),
                "rouge-2": round(r2, 4),
                "rouge-l": round(rl, 4)
            })

        # =========================
        # 전체 평균
        # =========================
        overall = {
            "filename": "ALL_FILES_AVERAGE",
            "rouge-1": round(sum(self.global_scores["r1"]) / len(self.global_scores["r1"]), 4),
            "rouge-2": round(sum(self.global_scores["r2"]) / len(self.global_scores["r2"]), 4),
            "rouge-l": round(sum(self.global_scores["rl"]) / len(self.global_scores["rl"]), 4),
            "file_count": len(self.global_scores["r1"])
        }

        results.insert(0, overall)

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n✅ Evaluation 완료")
        print(f"📂 저장 위치: {self.output_path}")