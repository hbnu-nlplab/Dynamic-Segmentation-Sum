from segmenter import MeetingSegmenter
from retriever import SegmentRetriever
from sub_summarizer import SubTopicSummarizer
from total_summarizer import TotalTopicSummarizer


class MeetingSummarizationPipeline:
    def __init__(self, model_name, openai_model, base_dir):
        self.paths = {
            "original": f"{base_dir}/test_dataset", # 회의록(팀벨) 데이터셋 경로
            "seg": f"{base_dir}/get_seg_merge",
            "related": f"{base_dir}/get_related_seg_win_10",
            "sub": f"{base_dir}/gen_sub_win_10",
            "total": f"{base_dir}/gen_total_win_10",
        }
        self.model_name = model_name
        self.openai_model = openai_model

    def run_all(self):
        print("\n🚀 [1/4] 세그먼트 분할 시작")
        MeetingSegmenter(self.model_name, self.paths["original"], self.paths["seg"]).run()

        print("\n🚀 [2/4] 세그먼트-토픽 매칭 시작")
        SegmentRetriever(self.model_name, self.paths["seg"], self.paths["original"], self.paths["related"]).run()

        print("\n🚀 [3/4] Sub-topic 요약 생성")
        SubTopicSummarizer(self.paths["related"], self.paths["sub"], self.openai_model).run()

        print("\n🚀 [4/4] Total summary 생성")
        TotalTopicSummarizer(self.paths["sub"], self.paths["original"], self.paths["total"], self.openai_model).run()

        print("\n🎉 전체 파이프라인 완료!")


if __name__ == "__main__":
    base_dir = "../Dynamic-Segmentation-Sum" # base 디렉 경로
    pipeline = MeetingSummarizationPipeline(
        model_name="Qwen/Qwen3-Embedding-8B",
        openai_model="gpt-4o-mini",
        base_dir=base_dir
    )
    pipeline.run_all()
