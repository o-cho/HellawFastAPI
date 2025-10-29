# services/agents/common_agents.py
import json, asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .searching import (
    summarize_context_for_search,
    hybrid_search,
    get_unique_docs,
    fetch_full_text,
)

def get_llm(model="gpt-4.1-mini", temperature=0.5):
    """공통 LLM 생성 함수"""
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        streaming=True
    )

async def stream_response(chain, inputs):
    """공통 스트리밍 처리"""
    accumulated = ""
    async for chunk in chain.astream(inputs):
        token = getattr(chunk, "content", str(chunk))
        accumulated += token
        yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.01)
    yield f"data: {json.dumps({'full': accumulated}, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"

async def free_chat_agent(query: str, domain: str, memory_context):
    """자유 질의 응답 메서드"""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        당신은 {domain} 분야를 포함한 다양한 법률 지식을 가진 전문 상담사입니다.
        사용자의 질문에 대해 법률 용어나 절차를 이해하기 쉽게 설명해주세요.

        - 너무 형식적인 문체는 피하고, 상담하듯 자연스럽게 답변하세요.
        - 필요시 참고할 만한 법 조항을 인용하되, 쉬운 용어 설명을 덧붙이세요.
        - 사용자에게 도움이 될 만한 구체적 행동 팁을 덧붙이면 좋습니다.
        - 지나치게 단정하지 말고, “일반적으로는 ~” 같은 표현을 사용하세요.
        - 지나치게 길게 말하지 마세요.
        """),
        MessagesPlaceholder(variable_name="history"),
        ("user", "새 사용자 발화: {query}")
    ])
    chain = prompt | llm
    history_vars = memory_context.load_memory_variables({})
    await asyncio.sleep(0)
    async for chunk in stream_response(chain, {
        "query": query,
        "domain": domain,
        "history": history_vars.get("history", [])
    }):
        yield chunk

domain_checklists = {
    "채권/금전거래": [
        "거래 또는 채권의 종류 (대여금, 투자금, 보증금 등)",
        "문제의 원인 (미상환, 지연, 이자 분쟁 등)",
        "피해 또는 손실의 정도"
    ],
    "부동산/임대차": [
        "계약 형태 (전세, 월세, 매매 등)",
        "문제 행위 또는 분쟁 원인 (보증금 미반환, 계약 해지 등)",
        "현재 결과 또는 손해 (거주 불가, 금전 손실 등)"
    ],
    "교통사고": [
        "사고 유형 (보행자, 차량 간, 자전거 등)",
        "사고 원인 또는 행위 (신호 위반, 과속, 부주의 등)",
        "피해 결과 (부상, 사망, 차량 파손 등)"
    ],
    "노동/고용": [
        "근로 형태 (정규직, 계약직, 프리랜서 등)",
        "문제 행위 또는 분쟁 원인 (부당해고, 임금체불, 계약불이행 등)",
        "피해 또는 결과 (임금 미지급, 복직 거부 등)"
    ],
    "의료사고": [
        "진료 또는 시술의 종류 (수술, 치료, 진단 등)",
        "문제의 원인 (과실, 오진, 부주의 등)",
        "피해 결과 (신체 손상, 후유증, 사망 등)"
    ]
}

async def info_gathering_agent(query: str, domain: str, memory_context):
    """사건 정보 수집 메서드"""
    llm = get_llm()
    checklist = domain_checklists.get(domain, ["상황 설명", "원인", "결과"])
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
            당신은 {domain} 분야의 전문 법률 상담사입니다.  
            사용자의 발화를 근거로 사건 파악을 위해 필요한 질문을 던지세요.
            공감하는 어투로 자연스럽게 이어가세요.
            
            판단 기준:
            - 아래 항목이 충족되어야 'ready_for_advice = true' 로 간주합니다.
            {checklist}
            - 항목이 전부 충족되지 않더라도, 조언을 하기에 충분한 정보가 모이면 'ready_for_advice = true'로 간주합니다.

            ---
            임무:
            1. 먼저 사용자의 상황을 2~3문장으로 요약하고 공감합니다.
            2. 다음 중 하나를 반드시 수행하세요:
                - 정보가 부족하면 추가 질문을 하세요.
                - 충분히 이해했다면 **반드시** 마지막 문장에 아래 문구를 포함하세요: "실제 판례를 검색 중입니다."
            3. 절대 이 문구를 생략하지 마세요. 이 문구는 다음 단계(판례 조언 전환)를 위한 신호입니다.

            """),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{query}")
    ])
    chain = prompt | llm
    history_vars = memory_context.load_memory_variables({})
    await asyncio.sleep(0)
    async for chunk in stream_response(chain, {
        "query": query,
        "history": history_vars.get("history", [])
    }):
        yield chunk

async def advising_agent(user_query: str, domain: str, memory_context:str):
    """판례 기반 조언"""
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        당신은 {domain} 분야의 전문 법률 조력자입니다.

        유사한 판례:
        {law_data}

        - 사용자의 상황과 가장 유사한 판례를 중심으로 조언을 제공합니다.
        - 먼저 판례의 내용을 간단하게 요약합니다.
        - 반드시 판례의 일부 문장을 인용해 근거를 제시하며, 인용한 판례의 doc_id를 명시합니다. (단, doc_id라는 용어를 사용하지 마세요.)
        - 400~500자 내외로 자연스럽게 작성합니다.
        - 판례가 상황과 다르면 인용하지 않습니다.
        """),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{user_query}")
    ])

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

    chain = prompt | llm
    history_vars = memory_context.load_memory_variables({})
    await asyncio.sleep(0)
    async for chunk in stream_response(chain, {
        "query": user_query,
        "law_data": law_data,
        "domain": domain,        
        "history": history_vars.get("history", [])
    }):
        yield chunk

async def guidance_agent(advice_text: str, domain: str, memory_context):
    """실행 조언"""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        당신은 {domain} 분야의 전문 법률 조력자입니다.

        직전 법률 조언:
        {advice_text}

        임무:
        - 사용자가 실제로 취할 수 있는 구체적인 행동 단계를 안내하세요.
        - 필요한 서류, 기관, 주의사항을 단계별로 명확히 정리하세요.
        - 마무리에 "궁금하신 부분이 있나요?"로 끝내세요.
        """),
        MessagesPlaceholder(variable_name="history")
    ])
    chain = prompt | llm
    history_vars = memory_context.load_memory_variables({})
    await asyncio.sleep(0)
    async for chunk in stream_response(chain, {
        "advice_text": advice_text,
        "domain": domain,
        "history": history_vars.get("history", [])
    }):
        yield chunk