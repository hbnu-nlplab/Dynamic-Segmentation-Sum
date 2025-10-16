import os
import json
import openai
from pathlib import Path
from dotenv import load_dotenv


class SubTopicSummarizer:
    def __init__(self, input_dir, output_dir, model="gpt-4o-mini"):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.input_dir, self.output_dir = Path(input_dir), Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = model

    def generate_summary(self, topic, sentences):
        text = " ".join(sentences)
        prompt = f"""
    당신은 회의 요약 전문가입니다.
    주제: {topic}
    문장들:
    {text}

    이 주제에 대한 요약을 명확하고 간결하게 4문장 내외로 작성하세요.
"""
        response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5, max_tokens=500
        )
        return response.choices[0].message.content.strip()

    def run(self):
        for file in sorted(self.input_dir.glob("*_topic_segments.json")):
            with open(file, "r", encoding="utf-8") as f:
                topic_segments = json.load(f)
            result = {}
            for topic, segs in topic_segments.items():
                sents = []
                for seg in sorted(segs, key=lambda x: x.get("rank", 999)):
                    sents.extend(seg.get("sentences", []))
                if not sents: continue
                summary = self.generate_summary(topic, sents)
                result[topic] = {"summary": summary, "count": len(sents)}
            out_path = self.output_dir / file.name.replace("_topic_segments.json", "_topic_summaries.json")
            json.dump(result, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            print(f"🧾 Sub-topic 요약 저장 완료: {out_path.name}")
