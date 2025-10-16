# Dynamic-Segmentation-Sum

pipeline.py 파일을 실행시키면 순서대로 document segmentation -> find relate segment -> sub topic sum -> total topic sum 이 수행됩니다.

pipeline.py 파일을 실행시켰을 때 나오는 output은 test_ouput에 저장해두었습니다!

경로 지정

    - pipeline.py
    
        - MeetingSummarizationPipeline/original 변수 -> dataset 경로
        - main/base_dir -> base 디렉 경로

open ai api를 사용할 때 key 지정

    - base 디렉에 .env파일을 만들어서 키값 설정하면 됩니다.
    
        - ex) OPENAI_API_KEY=sk-proj-LCf...
        
