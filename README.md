# ir_project
팀으로 진행(3명) -수정 각자 인사이트 공유 및 코드는 각자 구현  싱글턴 과학지식 180문항, 멀티턴 과학지식 20문항, 일상대화 20문항 총 220개의 문항 구성 prompt를 바탕으로 LLM이 과학 지식에 대해서는 rag 기능으로 api호출을 통해 Elasticsearch db에 질문, 일상 질문에 대해서는 생성 function_calling prompt를 통해 검색한 것을 바탕으로 persona_qa prompt를 바탕으로 최종답변을 생성한다.
