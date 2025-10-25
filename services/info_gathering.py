"""
사용자가 사례를 제시하면, 사용자의 상황을 파악하기 위해
필요한 정보를 수집하는 에이전트.
"""

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from openai import OpenAI
from config import ELASTIC_URL, ELASTIC_USER, ELASTIC_PASS, OPENAI_API_KEY
import json, re
import asyncio

# 환경 변수 로드
load_dotenv()

# 클라이언트 초기화
es = Elasticsearch(ELASTIC_URL, basic_auth=(ELASTIC_USER, ELASTIC_PASS), verify_certs=False)
client = OpenAI(api_key=OPENAI_API_KEY)

async def info_gathering_agent(query: str, domain: str, memory_context: str = ""):
    """
    사용자의 상황을 분석하고 필요한 정보를 수집하며,
    조언 단계로 넘어갈 준비 여부를 판단함.
    """

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
    사용자의 발화를 근거로, 사건 파악을 위해 필요한 질문을 던지거나
    충분한 정보가 모였는지를 판단하세요.

    ---
    지금까지의 대화:
    {memory_context}

    새 사용자 발화:
    "{query}"
    ---

    판단 기준:
    - 아래 항목이 모두 충족되어야 'ready_for_advice = true' 로 간주합니다.
      ({', '.join(checklist)})
    - 단 하나라도 불확실하거나 누락된 요소가 있다면 False로 판단합니다.
    - 단순히 피해 사실만 언급했을 경우에는 False입니다.
    - 거래 시점, 원인, 결과 중 어느 하나라도 모호하면 False입니다.

    ---
    임무:
    1. 우선 사용자의 상황을 2-3 문장 정도로 요약하고 공감해주세요.
    2. 다음 중 하나를 수행하세요:
       - 정보 부족 → 판단 기준을 바탕으로 추가 질문을 제시하세요.
       - 충분함 → 상담 톤으로 "실제 판례를 검색 중입니다."

    ---
    출력(JSON):
    {{
      "message": "사용자에게 보낼 자연스러운 문장",
      "ready_for_advice": true or false,
      "next_state": "info_gathering" 또는 "advising"
    }}
    """

    loop = asyncio.get_event_loop()
    res = await loop.run_in_executor(
        None,
        lambda: client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        ),
    )

    try:
        content = res.choices[0].message.content.strip()
        json_str = re.search(r"\{[\s\S]*\}", content)
        if json_str:
            return json.loads(json_str.group())
        else:
            raise ValueError("No JSON found in response")
    except Exception as e:
        print(f"JSON 파싱 오류: {e}")
        print(res.choices[0].message.content)
        return {
            "message": res.choices[0].message.content.strip(),
            "ready_for_advice": False,
            "next_state": "info_gathering",
        }
