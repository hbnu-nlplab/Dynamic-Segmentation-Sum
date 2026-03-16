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
You are an expert in generating a final meeting summary based on sub-topic-level summaries under a total_topic.
Below are meeting summaries, each related to the total_topic.
Your task is to generate a final summary related to the total_topic, using the sub_topic_summaries as reference.

- total_topic:
"{total_topic}"

- sub_topic_summaries:
{combined}
    
- Guidelines:
    - Integrate the content **naturally without redundancy**.
    - Write in a concise and clear style, similar to news articles.
    - Write **a single structurally organized summary** that reflects the overall topic.
    - The summary should be around 5~6 sentences.

✳️ Output Format:
    - Output only the final summary. Do not include any additional explanation or titles.
    - Write in Korean.
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
