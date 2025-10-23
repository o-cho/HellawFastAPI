from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
from config import ELASTIC_URL, ELASTIC_USER, ELASTIC_PASS, OPENAI_API_KEY
from .model_loader import get_model

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

# 하이브리드 검색 
def hybrid_search(query: str, domain_keyword: str, k: int = 5):

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

    keyword_result = es.search(index=INDEX_CHUNK, body=bm25_query)
    hits = keyword_result["hits"]["hits"]

    if not hits:
        print("[1차 키워드 기반 검색] 결과 없음")
        return []

    print(f"[1차 키워드 기반 검색] 문서 수: {len(hits)}")

    # [2차 의미 기반 검색] cosine 유사도를 기반으로 의미가 유사한 문서를 상위권으로 랭크.
    query_vector = model.encode(query)
    docs, embeddings = [], []

    for hit in hits:
        src = hit["_source"]
        vec = src.get(VECTOR_FIELD)
        if vec is not None:
            docs.append(src)
            embeddings.append(vec)

    embeddings = np.array(embeddings)
    scores = np.dot(embeddings, query_vector) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vector)
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
def fetch_full_text(doc_id):
    query = {"query": {"term": {"doc_id": doc_id}}}
    res = es.search(index=INDEX_FULL, body=query)

    if not res["hits"]["hits"]:
        print(f"⚠️ {doc_id} not found in {INDEX_FULL}")
        return None

    src = res["hits"]["hits"][0]["_source"]
    sentences = src.get("sentences")

    # sentences가 문자열인지 리스트인지 구분함.
    if isinstance(sentences, list):
        return "\n".join(sentences)
    elif isinstance(sentences, str):
        return sentences.strip()
    else:
        return None