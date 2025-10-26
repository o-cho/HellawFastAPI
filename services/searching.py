"""
RAG 검색기.
"""

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
from config import ELASTIC_URL, ELASTIC_USER, ELASTIC_PASS, OPENAI_API_KEY
from .model_loader import get_model
import asyncio

# 환경 변수 로드
load_dotenv()

# 클라이언트 초기화
es = Elasticsearch(ELASTIC_URL, basic_auth=(ELASTIC_USER, ELASTIC_PASS), verify_certs=False)
model = get_model()
client = OpenAI(api_key=OPENAI_API_KEY)

# 필요한 index 및 vector_field 선언
INDEX_CHUNK = "minsa_data"
INDEX_FULL = "minsa_judgement"
VECTOR_FIELD = "sentences_vector"

# 검색용 질의 요약기
# 전체 대화 맥락과 최신 질의를 분석, 판례 검색에 적합한 구체적 문장으로 변환함.
async def summarize_context_for_search(memory_context, latest_query):
    prompt = f"""
    당신은 법률 AI 상담사입니다.
    아래는 지금까지의 사용자와의 대화 내용입니다.

    {memory_context}

    사용자의 최신 발화: "{latest_query}"

    ## 임무
    1. 전체 대화를 바탕으로 사용자가 실제로 묻고 있는 법적 문제 상황을 1문장으로 요약하세요.
    2. 판결문 검색을 위해 구체적인 문장으로 표현해야 합니다.
       - 주체(누가): 예) 보행자, 운전자, 임대인, 근로자 등
       - 행위(무엇을 했는가): 예) 신호위반, 계약 위반, 부당해고 등
       - 결과(어떤 일이 발생했는가): 예) 교통사고 발생, 손해배상 청구 등
       - 쟁점(법적으로 알고 싶은 핵심): 예) 과실비율, 책임 범위, 손해액 산정 등
    3. 존칭이나 불필요한 문장은 넣지 말고, 구체적이지만 짧은 자연스러운 서술문 형태로 작성하세요.
    4. 검색 엔진이 이해하기 쉽도록 문어체를 사용하세요.

    ## 출력 형식 예시
    - 보행자가 빨간불에 횡단보도를 건너다 좌회전 차량과 충돌한 사고에서 과실비율 판단
    - 임차인이 월세를 연체하여 계약이 해지된 경우 보증금 반환 범위
    - 근로자가 정당한 이유 없이 해고된 경우 부당해고 인정 여부

    출력:
    """

    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return res.choices[0].message.content.strip()


# 하이브리드 검색 
async def hybrid_search(query: str, domain_keyword: str, k: int = 5):

    # [1차 키워드 기반 검색] domain_keyword와 사용자 입력 query를 바탕으로 키워드 기반 1차 검색.
    print(f"\n검색 시작: '{query}' | domain='{domain_keyword}'")

    bm25_query = {
        "size": 500,
        "query": {
            "bool": {
                "must": [
                    {"term": {"domain": domain_keyword}},
                    {"match": {"text": query}}
                ]
            }
        }
    }

    keyword_result = await asyncio.to_thread(
        es.search, index=INDEX_CHUNK, body=bm25_query
    )
    hits = keyword_result["hits"]["hits"]

    if not hits:
        print("[1차 키워드 기반 검색] 결과 없음")
        return []

    print(f"[1차 키워드 기반 검색] 문서 수: {len(hits)}")

    # [2차 의미 기반 검색] cosine 유사도를 기반으로 의미가 유사한 문서를 상위권으로 랭크.
    query_vector = await asyncio.to_thread(model.encode, query)

    docs, embeddings = [], []

    for hit in hits:
        src = hit["_source"]
        vec = src.get(VECTOR_FIELD)
        if vec is not None:
            docs.append(src)
            embeddings.append(vec)

    embeddings = np.array(embeddings)
    scores = await asyncio.to_thread(
        lambda: np.dot(embeddings, query_vector)
        / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vector))
    )

    top_indices = np.argsort(scores)[::-1]
    results = [(docs[i], float(scores[i])) for i in top_indices]

    print(f"[2차 의미 기반 검색] 문서 수: {len(results)}")
    return results

# 중복 doc_id 제거 및 상위 N개 선택
# 하나의 문서가 chunk 단위로 나눠져 있기 때문에 중복된 doc_id가 검색 결과로 매치될 수 있다.
# 가장 점수가 높은 chuck를 바탕으로 중복 id를 제거하고 상위 N개를 선택한다.
def get_unique_docs(results, top_n=3):
    seen_ids = set()
    unique_results = []

    for doc, score in results:
        doc_id = doc.get("doc_id")
        if doc_id and doc_id not in seen_ids:
            seen_ids.add(doc_id)
            unique_results.append((doc, score))
        if len(unique_results) >= top_n:
            break

    print(f"고유 doc_id {len(unique_results)}개 선택 완료: {seen_ids}")
    return unique_results

# 원문 조회
# chuck에서 가져온 doc_id를 바탕으로 판결문 원문을 조회한다.
async def fetch_full_text(doc_id):
    query = {"query": {"term": {"doc_id": doc_id}}}
    res = await asyncio.to_thread(es.search, index=INDEX_FULL, body=query)

    if not res["hits"]["hits"]:
        print(f"{doc_id} not found in {INDEX_FULL}")
        return None

    src = res["hits"]["hits"][0]["_source"]
    sentences = src.get("sentences")

    if isinstance(sentences, list):
        return "\n".join(sentences)
    elif isinstance(sentences, str):
        return sentences.strip()
    else:
        return None
