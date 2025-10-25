"""
RAG 검색을 통해 사용자의 사례와 유사한 판례를 검색하고, 
관련된 조언을 제공하는 에이전트.
"""

from .searching import hybrid_search, get_unique_docs, fetch_full_text, summarize_context_for_search
from .model_loader import get_model
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
from config import ELASTIC_URL, ELASTIC_USER, ELASTIC_PASS, OPENAI_API_KEY
import asyncio

# 환경 변수 로드
load_dotenv()

# 클라이언트 초기화
es = Elasticsearch(ELASTIC_URL, basic_auth=(ELASTIC_USER, ELASTIC_PASS), verify_certs=False)
model = get_model()
client = OpenAI(api_key=OPENAI_API_KEY)

async def advising_agent(query, memory_context, domain):
    case_context = f"이전 대화 내용:\n{memory_context}\n\n사용자 질의:\n{query}"

    # 검색 쿼리 생성
    search_query = await summarize_context_for_search(memory_context, query)
    print(f"생성된 검색 질의문: {search_query}")

    # 검색 실행
    results = await hybrid_search(search_query, domain_keyword=domain, k=5)
    unique_docs = get_unique_docs(results, top_n=3)
    if not unique_docs:
        return {"status": "error", "message": "관련 판례를 찾지 못했습니다."}

    # 판결문 수집
    doc_list = []
    for doc, score in unique_docs:
        doc_id = doc.get("doc_id")
        full_text = await fetch_full_text(doc_id)
        if not full_text:
            continue
        doc_list.append({
            "doc_id": doc_id,
            "case_name": doc.get("case_name", "제목없음"),
            "score": round(score, 3),
            "excerpt": full_text[:800] + "..." if len(full_text) > 800 else full_text
        })

    docs_text = '\n\n'.join(
        [f"{d['case_name']} ({d['doc_id']})\n{d['excerpt']}" for d in doc_list]
        )

    prompt = f"""
    당신은 법률 전문 AI 조언가입니다.
    다음은 사용자와의 대화 요약 및 질의입니다:

    {case_context}

    아래는 참고할 수 있는 판결문입니다:
    {docs_text}

    ## 임무
    - 사용자의 상황과 가장 유사한 판례를 중심으로 조언을 제공합니다.
    - 먼저 판례의 내용을 간단하게 요약합니다.
    - 반드시 판례의 일부 문장을 인용해 근거를 제시하며, 인용한 판례의 doc_id를 명시합니다. (단, doc_id라는 용어를 사용하지 마세요.)
    - 400~500자 내외로 자연스럽게 작성합니다.
    - 판례가 상황과 다르면 인용하지 않습니다.

    법률 조언:
    """

    loop = asyncio.get_event_loop()
    res = await loop.run_in_executor(
        None,
        lambda: client.chat.completions.create(
            model="gpt-4o-mini",  # 더 빠른 모델
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
    )

    advice_text = res.choices[0].message.content.strip()

    return {
        "status": "ok",
        "query": query,
        "search_query": search_query,
        "domain": domain,
        "advice": advice_text,
        "references": doc_list
    }
