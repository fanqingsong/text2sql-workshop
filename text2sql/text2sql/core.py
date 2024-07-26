from langchain.chat_models import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from pydantic import BaseModel
from langchain_core.runnables.base import Runnable
import pydantic

class Text2SQL(BaseModel):

    uri: str = "postgresql://postgres:changeme@localhost:5432"
    # model: str = "gpt-4-1106-preview"
    model: str = "internlm/internlm2_5-7b-chat-gguf/internlm2_5-7b-chat-q2_k.gguf"
    
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
            api_key="...",  
            base_url="http://192.168.0.108:1234/v1/",
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