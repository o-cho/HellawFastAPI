from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from config import OPENAI_API_KEY
import json, asyncio

# OPENAI 모델 정의
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.5,
    streaming=True
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
    당신은 {domain} 분야를 포함한 다양한 법률 지식을 가진 전문 상담사입니다.
    사용자의 질문에 대해 법률 용어나 절차를 이해하기 쉽게 설명해주세요.

    - 너무 형식적인 문체는 피하고, 상담하듯 자연스럽게 답변하세요.
    - 필요시 참고할 만한 법 조항을 인용하되, 쉬운 용어 설명을 덧붙이세요.
    - 사용자에게 도움이 될만한 구체적 행동 팁을 덧붙이면 좋습니다.
    - 지나치게 단정하지 말고, “일반적으로는 ~” 같은 표현을 사용하세요.
    """),
    MessagesPlaceholder(variable_name="history"),
    ("user", "새 사용자 발화: {query}")
])



async def free_chat_agent(query: str, domain: str, memory_context:str):
    """
    법률 관련 질문에 전반적으로 응답해줌.
    """
    # llm 체인 구성
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory_context)

    accumulated = ""

    async for chunk in chain.astream({
        "query":query
    }):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        accumulated += token
        yield f"data: {json.dumps({'token':token}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.01)
    
    # 최종 데이터 출력
    yield f"data: {json.dumps({'full': accumulated}, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"