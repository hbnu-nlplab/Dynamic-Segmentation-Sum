from segmenter import MeetingSegmenter
from retriever import SegmentRetriever
from sub_summarizer import SubTopicSummarizer
from total_summarizer import TotalTopicSummarizer
from evaluator import SummaryEvaluator

class MeetingSummarizationPipeline:
    def __init__(self, model_name, openai_model, base_dir):
        self.paths = {
            "original": f"{base_dir}/test_dataset", # 회의록(팀벨) 데이터셋 경로
            "seg": f"{base_dir}/output/semantic_segmentation",
            "related": f"{base_dir}/output/relevant_seg",
            "sub": f"{base_dir}/output/gen_sub",
            "total": f"{base_dir}/output/gen_total",
            "eval": f"{base_dir}/output/evaluation/result.json",
        }
        self.model_name = model_name
        self.openai_model = openai_model

    def run_all(self):
        print("\n🚀 [1/5] 세그먼트 분할 시작")
        MeetingSegmenter(self.model_name, self.paths["original"], self.paths["seg"]).run()

        print("\n🚀 [2/5] 세그먼트-토픽 매칭 시작")
        SegmentRetriever(self.model_name, self.paths["seg"], self.paths["original"], self.paths["related"]).run()

        print("\n🚀 [3/5] Sub-topic 요약 생성")
        SubTopicSummarizer(self.paths["related"], self.paths["sub"], self.openai_model).run()

        print("\n🚀 [4/5] Total summary 생성")
        TotalTopicSummarizer(self.paths["sub"], self.paths["original"], self.paths["total"], self.openai_model).run()

        print("\n🚀 [5/5] Evaluation 시작")
        SummaryEvaluator(self.paths["total"], self.paths["original"], self.paths["eval"]).run()
        print("\n🎉 전체 파이프라인 완료!")


if __name__ == "__main__":
    base_dir = "/home/kilab_ndw/etri_sum/NEW_BORN_embed/ko/qwen/Dynamic-Segmentation-Sum" # base 디렉 경로
    pipeline = MeetingSummarizationPipeline(
        model_name="Qwen/Qwen3-Embedding-8B",
        openai_model="gpt-4o-mini",
        base_dir=base_dir
    )
    pipeline.run_all()
