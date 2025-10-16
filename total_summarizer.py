import os
import json
import openai
from pathlib import Path
from dotenv import load_dotenv


class TotalTopicSummarizer:
    def __init__(self, input_dir, gt_dir, output_dir, model="gpt-4o-mini"):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.input_dir, self.gt_dir, self.output_dir = Path(input_dir), Path(gt_dir), Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = model

    def run(self):
        for file in sorted(self.input_dir.glob("*_topic_summaries.json")):
            base = file.stem.replace("_topic_summaries", "")
            gt_file = self.gt_dir / f"{base}.json"
            if not gt_file.exists(): continue

            with open(file, "r", encoding="utf-8") as f:
                subs = json.load(f)
            with open(gt_file, "r", encoding="utf-8") as f:
                gt = json.load(f)

            total_topic = gt.get("total_summary", [])[0].get("total_topic", "")
            combined = "\n".join(f"- {d['summary']}" for d in subs.values() if d["summary"])
            if not combined:
                continue

            prompt = f"""
    당신은 회의 전체 내용을 요약하는 전문가입니다.
    아래의 sub-topic 요약문들을 참고하여 '{total_topic}'에 대한 전체 요약을 4문장 내외로 작성하세요.
    {combined}
"""
            response = openai.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5, max_tokens=1000
            )
            summary = response.choices[0].message.content.strip()
            out_path = self.output_dir / f"{base}_total_generated_summary.json"
            json.dump({"total_topic": total_topic, "summary": summary},
                      open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            print(f"✅ Total summary 생성 완료: {base}")
