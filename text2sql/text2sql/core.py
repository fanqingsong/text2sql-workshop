from langchain.chat_models import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.agent_types import AgentType
from sqlalchemy import create_engine
from pydantic import BaseModel
from langchain_core.runnables.base import Runnable
import pydantic

import os


OPENAI_API_KEY=os.environ['OPENAI_API_KEY']
ZHIPUAI_API_KEY=os.environ['ZHIPUAI_API_KEY']
SILICONFLOWAI_API_KEY=os.environ['SILICONFLOWAI_API_KEY']


class Text2SQL(BaseModel):

    uri: str = "postgresql://postgres:changeme@localhost:5432/postgres"
    model: str = "gpt-4-1106-preview"
    # model: str = "internlm/internlm2_5-7b-chat-gguf/internlm2_5-7b-chat-q2_k.gguf"
    # model: str = "glm-4-flash"

    # model: str = "Qwen/Qwen2.5-7B-Instruct"
    # model: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" 
    # model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    # model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    # model: str = "THUDM/glm-4-9b-chat"
    # model: str = "Qwen/Qwen2.5-Coder-32B-Instruct"
    # model: str = "Pro/deepseek-ai/DeepSeek-V3"
    # model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"



    temperature: int = 0

    class Config:
        arbitrary_types_allowed = True

    @pydantic.computed_field()
    @property
    def db(self) -> SQLDatabase:
        return SQLDatabase(engine = create_engine(self.uri))
    
    @pydantic.computed_field()
    @property
    def llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model = self.model,
            temperature = self.temperature,
            api_key=OPENAI_API_KEY,  
            # api_key=ZHIPUAI_API_KEY,  
            # base_url="https://open.bigmodel.cn/api/paas/v4/",

            # openai_api_key=SILICONFLOWAI_API_KEY,
            # openai_api_base="https://api.siliconflow.cn/v1"
        )
    
    @pydantic.computed_field()
    @property
    def chain(self) -> Runnable:
        return create_sql_query_chain(
            llm = self.llm,
            db = self.db
        )
    
    def query(self, question: str):

        response = self.chain.invoke({"question": question})
        sql_query = response.split("SQLQuery:")[0]

        return sql_query
    
    def run_sql(self, sql_query: str):
        return self.db.run(sql_query)

    def run(self, question: str):
        agent_executor = create_sql_agent(self.llm, 
                                          db=self.db, 
                                        #   agent_type="tool-calling",
                                          agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                                        #   toolkit=SQLDatabaseToolkit(db=self.db, llm=self.llm), 
                                          verbose=True)
        
        return agent_executor.invoke(question)
