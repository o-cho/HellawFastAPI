from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from .searching import (
    summarize_context_for_search,
    hybrid_search,
    get_unique_docs,
    fetch_full_text,
) 
import json, asyncio


# OPENAI 모델 정의
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.5,
    streaming=True
)
 
prompt = ChatPromptTemplate.from_messages([
    ("system", """
     당신은 {domain} 분야의 전문 법률 조력자입니다.

     유사한 판례 : {law_data}

    - 사용자의 상황과 가장 유사한 판례를 중심으로 조언을 제공합니다.
    - 먼저 판례의 내용을 간단하게 요약합니다.
    - 반드시 판례의 일부 문장을 인용해 근거를 제시하며, 인용한 판례의 doc_id를 명시합니다. (단, doc_id라는 용어를 사용하지 마세요.)
    - 400~500자 내외로 자연스럽게 작성합니다.
    - 판례가 상황과 다르면 인용하지 않습니다.
    """),
    MessagesPlaceholder(variable_name="history"),
    ("user", "새 사용자 발화: {query}")
])

# # 메모리 지정
# memory = ConversationBufferMemory(memory_key = "history", input_key="query", return_messages=True)

async def advising_agent(user_query:str, domain:str, memory_context:str):
    """
    판례를 기반으로 조언을 제공함.
    """

    # llm 체인 구성
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory_context)

    
    # 요약 문장 생성
    summary = await summarize_context_for_search(memory_context, user_query)
    print(f"검색 요약: {summary}")

    # RAG 검색
    results = await hybrid_search(summary, domain)
    if not results:
        yield f"data: {json.dumps({'token': '관련된 판례를 찾지 못했습니다.'}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        return

    # 중복 제거 후 상위 3개만 추출
    unique_docs = get_unique_docs(results, top_n = 3)

    # 판례 원문 가져오기
    full_texts = []
    
    for doc, score in unique_docs: 
        text = await fetch_full_text(doc["doc_id"])
        if text:
            full_texts.append({
                "doc_id": doc["doc_id"],
                "score": score,
                "text": text[:1200]
            })

    # 판례 텍스트 합치기
    law_data = "\n\n".join([
        f"[사례 {i+1}] ({d['score']})\n{d['text']}"
        for i, d in enumerate(full_texts)
    ])

    print(f"[검색된 판례 수] {len(full_texts)}개")

    # chain에 전달할 입력 데이터 준비
    inputs = {
        "query": user_query,
        "law_data": law_data,
        "domain": domain
    }

    accumulated = ""

    async for chunk in chain.astream(inputs):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        accumulated += token
        yield f"data: {json.dumps({'token':token}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.01)
    
    # 최종 데이터 출력
    yield f"data: {json.dumps({'full': accumulated}, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"