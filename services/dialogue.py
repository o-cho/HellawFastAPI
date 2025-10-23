import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from openai import OpenAI
from config import ELASTIC_URL, ELASTIC_USER, ELASTIC_PASS, OPENAI_API_KEY
from collections import deque
import json
import uuid

# 환경 변수 로드
load_dotenv()

# 클라이언트 초기화
es = Elasticsearch(ELASTIC_URL, basic_auth=(ELASTIC_USER, ELASTIC_PASS), verify_certs=False)
client = OpenAI(api_key=OPENAI_API_KEY)

# 메모리
class MemoryManager:
    # 10턴 초과 시 자동으로 오래된 대화 삭제.
    def __init__(self, max_turns=10):
        self.history = deque(maxlen=max_turns)
        self.conv_idx = str(uuid.uuid4()) # 세션 고유 아이디 생성
    
    # 대화 한 턴을 메모리에 추가함.
    def add(self, role, content):
        self.history.append({"role": role, "content": content})
    
    # 저장된 대화 내용을 개행으로 연결하여 하나의 문자열로 반환함.
    def get_context(self):
        return "\n".join(
            [f"{h['role'].upper()}: {h['content']}" for h in self.history]
        )
    
    # 새 대화 시작. 기존 기록을 초기화하고 conv_id를 재발급함.
    def reset(self):
        self.history.clear()
        self.conv_idx = str(uuid.uuid4())

async def dialogue_agent(query, memory_context, domain):
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

    checklist = domain_checklists.get(domain, ["상황 설명", "원인", "결과"])

    prompt = f"""
        당신은 {domain} 분야의 전문 법률 상담사입니다.  
        사용자와의 대화를 바탕으로 필요하면 질문을, 충분하면 조언을 제시하세요.

        ---

        ### 지금까지의 대화
        {memory_context}

        ### 새 사용자 발화
        "{query}"

        ---

        ### 판단 기준
        - 다음 항목이 모두 충족되면 'ready_for_advice = true'로 간주합니다.
        ({', '.join(checklist)})
        - 정보가 대부분 확보되었거나, 상황의 핵심이 이미 드러난 경우에도 True로 판단하세요.
        - 단순한 부가정보(금액, 세부 시점 등)가 없더라도 법적 판단이 가능하면 True로 하세요.

        ---

        ### 임무
        1. 사용자의 상황을 짧게 요약하세요.
        2. 다음 중 하나를 수행하세요:
        - 정보 부족 → 핵심 사실을 파악하기 위한 추가 질문 작성
        - 충분함 → 조언이 가능한 단계로 판단하고, 그 사실을 자연스럽게 언급
            (예: “이제 법적 판단을 도와드릴 수 있을 것 같습니다.”)
        3. **'ready_for_advice'가 True일 경우 실제 조언은 하지 마세요.**
        대신 “법률적 판단이 가능한 상황”임을 표시하는 멘트를 포함하세요.
        (조언은 이후 별도의 모듈이 담당합니다.)

        ---

        ### 출력(JSON만)
        {{
        "message": "사용자에게 보낼 자연스러운 말",
        "ready_for_advice": true or false
        }}
        """

    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )

    try:
        return json.loads(res.choices[0].message.content)
    except Exception as e:
        print(f"⚠️ JSON 파싱 오류: {e}")
        print(res.choices[0].message.content)
        # fallback
        return {"message": res.choices[0].message.content, "ready_for_advice": False}


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
