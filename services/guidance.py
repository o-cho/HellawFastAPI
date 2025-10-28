from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
import json, asyncio

# OPENAI 모델 정의
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.5,
    streaming=True
)

# TODO 직전 법률 조언을 넣을 필요가 있는지 확인하기. 
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


async def guidance_agent(advice_text: str, domain: str, memory_context:str):
    """
    사용자가 실제로 실천할 수 있는 법률적 조언을 제시함.
    """
    # llm 체인 구성
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory_context)

    accumulated = ""

    async for chunk in chain.astream({
        "advice_text":advice_text,
        "domain":domain
    }):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        accumulated += token
        yield f"data: {json.dumps({'token':token}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.01)
    
    # 최종 데이터 출력
    yield f"data: {json.dumps({'full': accumulated}, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"