from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from langchain_cohere import ChatCohere
from langchain.agents import AgentType, initialize_agent, load_tools
from dotenv import load_dotenv

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

# Set environment variables for API keys
load_dotenv('.env')


def search_query(query: str):
    custom_prompt = (
        "You are a knowledgeable assistant. When asked about any topic, provide a detailed and comprehensive response. "
        "Include background information, key points, significant achievements, relevant current events, and any controversies. "
        "Ensure your answer is well-rounded and informative."
    )

    llm = ChatCohere(model='command-r-plus', temperature=0)
    tools = load_tools(["serpapi" , 'openweathermap-api'], llm)

    agent_chain = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        prompt=custom_prompt
    )
    result = agent_chain.run(query)
    return result


@app.post("/search")
async def search(query: QueryRequest):
    try:
        result = search_query(query.query)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
