import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
import gradio as gr

from dotenv import load_dotenv
load_dotenv('config.env')

async def main():
    client = MultiServerMCPClient(
        {
            "mcp_server": {
                "transport": "sse",
                "url": "https://abidlabs-mcp-tool-http.hf.space/gradio_api/mcp/sse"
            },
        }
    )
    tools = await client.get_tools()

    # Create and run the agent
    agent = create_react_agent("google_genai:gemini-2.0-flash-lite", tools)
    return agent

agent = asyncio.run(main())

async def get_response(message, history):
    r = await agent.ainvoke({"messages": message})
    if r:
        response = r['messages'][-1].content
    return str(response)

demo = gr.ChatInterface(
    fn=get_response,
    type="messages",
    examples=["Prime factorization of 68"],
    title="Agent with MCP Tools",
    description="This is a simple agent that uses MCP tools to answer questions."
    )

demo.launch()