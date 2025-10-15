import pytest
import os
import json
import re
import datetime

from bridgic.core.model.types import *
from bridgic.core.model import *
from bridgic.core.utils.console import printer
from bridgic.llms.vllm.vllm_server_llm import VllmServerLlm

_api_base = os.environ.get("VLLM_SERVER_API_BASE")
_api_key = os.environ.get("VLLM_SERVER_API_KEY")
_model_name = os.environ.get("VLLM_SERVER_MODEL_NAME")

@pytest.fixture
def llm():
    llm = VllmServerLlm(
        api_base=_api_base,
        api_key=_api_key,
        timeout=5,
    )
    state_dict = llm.dump_to_dict()
    del llm
    llm = VllmServerLlm.__new__(VllmServerLlm)
    llm.load_from_dict(state_dict)
    return llm

@pytest.fixture
def date():
    date = datetime.datetime.now().strftime('%Y-%m-%d')
    return date

@pytest.fixture
def datetime_obj():
    datetime_obj = datetime.datetime.now()
    return datetime_obj

@pytest.fixture
def tools():
    class GetWeather(BaseModel):
        city: str = Field(description="The city to get the weather of.")
    class GetNews(BaseModel):
        date: datetime.date = Field(description="The date to get the news of.")
        topic: str = Field(description="The topic to get the news of.")
    tools = [
        Tool(name="get_weather", description="Get the weather of a city.", parameters=GetWeather.model_json_schema()),
        Tool(name="get_news", description="Get the news of a date and a topic.", parameters=GetNews.model_json_schema()),
    ]
    return tools

@pytest.fixture
def weather_messages(date):
    messages = [
        Message.from_text(
            text="You are a helpful assistant. You are good at calling the provided tools to solve problems.",
            role=Role.SYSTEM,
        ),
        Message.from_text(
            text="The weather of London.",
            role=Role.USER
        ),
        Message.from_tool_call(
            tool_calls=ToolCall(
                id="tool_123",
                name="get_weather",
                arguments={"city": "London"},
            ),
        ),
        Message.from_tool_result(
            tool_id="tool_123",
            content="London, 22°C, sunny with light clouds",
        ),
        Message.from_text(
            text="The weather in London is 22°C and sunny with light clouds.",
            role=Role.AI,
        ),
        Message.from_text(
            text=f"Thanks! Tell me the weather of Tokyo and today's sports news. Remember today is {date}.",
            role=Role.USER,
        ),
    ]
    return messages

def handle_response(date, response: Tuple[List[ToolCall], Optional[str]]):
    assert len(response) == 2
    tool_calls, content = response
    if tool_calls:
        for tool_call in tool_calls:
            printer.print(json.dumps(tool_call.model_dump()), color='purple')
            if tool_call.name == "get_weather":
                assert tool_call.arguments["city"] == "Tokyo"
            if tool_call.name == "get_news":
                assert tool_call.arguments["date"] == date
                assert len(tool_call.arguments["topic"]) > 0
    if content:
        printer.print(content, color='yellow')

@pytest.mark.skipif(
    (_api_key is None) or (_api_base is None) or (_model_name is None),
    reason="VLLM_SERVER_API_KEY or VLLM_SERVER_API_BASE or VLLM_SERVER_MODEL_NAME is not set",
)
def test_vllm_server_chat(llm):
    response = llm.chat(
        model=_model_name,
        messages=[Message.from_text(text="Hello, how are you?", role=Role.USER)],
    )
    printer.print(response)
    assert response.message.role == Role.AI
    assert response.message.content is not None

@pytest.mark.skipif(
    (_api_key is None) or (_api_base is None) or (_model_name is None),
    reason="VLLM_SERVER_API_KEY or VLLM_SERVER_API_BASE or VLLM_SERVER_MODEL_NAME is not set",
)
def test_vllm_server_stream(llm):
    response = llm.stream(
        model=_model_name,
        messages=[Message.from_text(text="Hello, how are you?", role=Role.USER)],
    )
    result = ""
    for chunk in response:
        result += chunk.delta
        assert chunk.delta is not None
        assert chunk.raw is not None
    assert len(result) > 0

@pytest.mark.skipif(
    (_api_key is None) or (_api_base is None) or (_model_name is None),
    reason="VLLM_SERVER_API_KEY or VLLM_SERVER_API_BASE or VLLM_SERVER_MODEL_NAME is not set",
)
@pytest.mark.asyncio
async def test_vllm_server_achat(llm):
    response = await llm.achat(
        model=_model_name,
        messages=[Message.from_text(text="Hello, how are you?", role=Role.USER)],
    )
    printer.print(response)
    assert response.message.role == Role.AI
    assert response.message.content is not None

@pytest.mark.skipif(
    (_api_key is None) or (_api_base is None) or (_model_name is None),
    reason="VLLM_SERVER_API_KEY or VLLM_SERVER_API_BASE or VLLM_SERVER_MODEL_NAME is not set",
)
@pytest.mark.asyncio
async def test_vllm_server_astream(llm):
    response = llm.astream(
        model=_model_name,
        messages=[Message.from_text(text="Hello, how are you?", role=Role.USER)],
    )
    result = ""
    async for chunk in response:
        result += chunk.delta
        assert chunk.delta is not None
        assert chunk.raw is not None
    assert len(result) > 0

@pytest.mark.skipif(
    (_api_key is None) or (_api_base is None) or (_model_name is None),
    reason="VLLM_SERVER_API_KEY or VLLM_SERVER_API_BASE or VLLM_SERVER_MODEL_NAME is not set",
)
def test_vllm_server_structured_output_pydantic_model(llm):
    class ThinkAndAnswer(BaseModel):
        thought: str = Field(description="The thought about the problem.", max_length=100)
        answer: str = Field(description="The answer to the question.", min_length=10)

    response: ThinkAndAnswer = llm.structured_output(
        model=_model_name,
        constraint=PydanticModel(model=ThinkAndAnswer),
        messages=[
            Message.from_text(
                text="You are a helpful assistant.",
                role=Role.SYSTEM,
            ),
            Message.from_text(
                text='''
What is the meaning of life? Please think about this question and give a short answer. 
Your answer should be in JSON format and should contain two keys: "thought" and "answer".
Don't think for long time. Don't answer in many words.
''',
                role=Role.USER,
            ),
        ],
    )
    printer.print("\n" + response.model_dump_json(), color='purple')
    assert response.thought is not None
    assert len(response.answer) >= 5

@pytest.mark.skipif(
    (_api_key is None) or (_api_base is None) or (_model_name is None),
    reason="VLLM_SERVER_API_KEY or VLLM_SERVER_API_BASE or VLLM_SERVER_MODEL_NAME is not set",
)
@pytest.mark.asyncio
async def test_vllm_server_astructured_output_pydantic_model(llm):
    class ThinkAndAnswer(BaseModel):
        thought: str = Field(description="The thought about the problem.", max_length=100)
        answer: str = Field(description="The answer to the question.", min_length=10)

    response: ThinkAndAnswer = await llm.astructured_output(
        model=_model_name,
        constraint=PydanticModel(model=ThinkAndAnswer),
        messages=[
            Message.from_text(
                text="You are a helpful assistant.",
                role=Role.SYSTEM,
            ),
            Message.from_text(
                text='''
What is the meaning of life? Please think about this question and give a short answer. 
Your answer should be in JSON format and should contain two keys: "thought" and "answer".
Don't think for long time. Don't answer in many words.
''',
                role=Role.USER,
            ),
        ],
    )
    printer.print("\n" + response.model_dump_json(), color='purple')
    assert response.thought is not None
    assert len(response.answer) >= 5

@pytest.mark.skipif(
    (_api_key is None) or (_api_base is None) or (_model_name is None),
    reason="VLLM_SERVER_API_KEY or VLLM_SERVER_API_BASE or VLLM_SERVER_MODEL_NAME is not set",
)
def test_vllm_server_structured_output_json_schema(llm):
    schema = {
        "type": "object",
        "properties": {
            "thought": {"type": "string", "description": "The thought about the problem.", "maxLength": 100},
            "answer": {"type": "string", "description": "The answer to the question.", "minLength": 5},
        },
        "required": ["thought", "answer"],
    }

    response: Dict[str, Any] = llm.structured_output(
        model=_model_name,
        constraint=JsonSchema(name="ThinkAndAnswer", schema_dict=schema),
        messages=[
            Message.from_text(
                text="You are a helpful assistant.",
                role=Role.SYSTEM,
            ),
            Message.from_text(
                text="What is the result of 12345 + 54321?",
                role=Role.USER,
            ),
        ],
    )
    printer.print("\n" + json.dumps(response), color='purple')
    assert isinstance(response["thought"], str)
    assert isinstance(response["answer"], str)

@pytest.mark.skipif(
    (_api_key is None) or (_api_base is None) or (_model_name is None),
    reason="VLLM_SERVER_API_KEY or VLLM_SERVER_API_BASE or VLLM_SERVER_MODEL_NAME is not set",
)
@pytest.mark.asyncio
async def test_vllm_server_astructured_output_json_schema(llm):
    schema = {
        "type": "object",
        "properties": {
            "thought": {"type": "string", "description": "The thought about the problem.", "maxLength": 100},
            "answer": {"type": "string", "description": "The answer to the question.", "minLength": 5},
        },
        "required": ["thought", "answer"],
    }

    response: Dict[str, Any] = await llm.astructured_output(
        model=_model_name,
        constraint=JsonSchema(name="ThinkAndAnswer", schema_dict=schema),
        messages=[
            Message.from_text(
                text="You are a helpful assistant.",
                role=Role.SYSTEM,
            ),
            Message.from_text(
                text="What is the result of 12345 + 54321?",
                role=Role.USER,
            ),
        ],
    )
    printer.print("\n" + json.dumps(response), color='purple')
    assert isinstance(response["thought"], str)
    assert isinstance(response["answer"], str)

@pytest.mark.skipif(
    (_api_key is None) or (_api_base is None) or (_model_name is None),
    reason="VLLM_SERVER_API_KEY or VLLM_SERVER_API_BASE or VLLM_SERVER_MODEL_NAME is not set",
)
def test_vllm_server_structured_output_ebnf_grammar(llm):
    ebnf_syntax = """
root ::= select_statement
select_statement ::= "select " columns " from " table " where " conditions
columns ::= (column (", " column)*) | "*"
column ::= "year" | "sales"
table ::= "sales_table"
conditions ::= condition (" and " condition)*
condition ::= column "=" number
number ::= "2020" | "2021" | "2022" | "2023" | "2024"
"""
    response: str = llm.structured_output(
        model=_model_name,
        constraint=EbnfGrammar(syntax=ebnf_syntax, description="A grammar for selecting sales data."),
        messages=[
            Message.from_text(
                text="You are a helpful assistant. You are good at writting SQL statements.",
                role=Role.SYSTEM,
            ),
            Message.from_text(
                text="""
Here is a table (table name: sales_table) about the sales data:
| id | year | sales |
| -- | ---- | ----- |
| 1  | 2021 | 20000 |
| 2  | 2022 | 20000 |
| 3  | 2023 | 30000 |
| 4  | 2024 | 40000 |
Write a SQL to select the sales of year of 2023 from the table.
""",
                role=Role.USER,
            ),
        ],
    )
    printer.print("\n" + response, color='purple')
    assert response == "select sales from sales_table where year=2023"

@pytest.mark.skipif(
    (_api_key is None) or (_api_base is None) or (_model_name is None),
    reason="VLLM_SERVER_API_KEY or VLLM_SERVER_API_BASE or VLLM_SERVER_MODEL_NAME is not set",
)
@pytest.mark.asyncio
async def test_vllm_server_astructured_output_ebnf_grammar(llm):
    ebnf_syntax = """
root ::= select_statement
select_statement ::= "select " columns " from " table " where " conditions
columns ::= (column (", " column)*) | "*"
column ::= "year" | "sales"
table ::= "sales_table"
conditions ::= condition (" and " condition)*
condition ::= column "=" number
number ::= "2020" | "2021" | "2022" | "2023" | "2024"
"""
    response: str = await llm.astructured_output(
        model=_model_name,
        constraint=EbnfGrammar(syntax=ebnf_syntax, description="A grammar for selecting sales data."),
        messages=[
            Message.from_text(
                text="You are a helpful assistant. You are good at writting SQL statements.",
                role=Role.SYSTEM,
            ),
            Message.from_text(
                text="""
Here is a table (table name: sales_table) about the sales data:
| id | year | sales |
| -- | ---- | ----- |
| 1  | 2021 | 20000 |
| 2  | 2022 | 20000 |
| 3  | 2023 | 30000 |
| 4  | 2024 | 40000 |
Write a SQL to select the sales of year of 2023 from the table.
""",
                role=Role.USER,
            ),
        ],
    )
    printer.print("\n" + response, color='purple')
    assert response == "select sales from sales_table where year=2023"

@pytest.mark.skipif(
    (_api_key is None) or (_api_base is None) or (_model_name is None),
    reason="VLLM_SERVER_API_KEY or VLLM_SERVER_API_BASE or VLLM_SERVER_MODEL_NAME is not set",
)
def test_vllm_server_structured_output_regex_email(llm):
    pattern = rf"^Emails:\n(- {RegexPattern.EMAIL.pattern})(\n(- {RegexPattern.EMAIL.pattern}))*$"
    response: str = llm.structured_output(
        model=_model_name,
        constraint=Regex(pattern=pattern, description="A regex for matching email addresses."),
        messages=[
            Message.from_text(
                text="You are a helpful assistant. You are good at extracting email addresses from text.",
                role=Role.SYSTEM,
            ),
            Message.from_text(
                text="Email addresses: jack@gmail.com and david@gmail.com and john@gmail.com",
                role=Role.USER,
            ),
        ],
    )
    printer.print("\n" + response, color='purple')
    emails = re.findall(pattern=r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", string=response)
    assert len(emails) == 3
    assert emails[0] == "jack@gmail.com"
    assert emails[1] == "david@gmail.com"
    assert emails[2] == "john@gmail.com"

@pytest.mark.skipif(
    (_api_key is None) or (_api_base is None) or (_model_name is None),
    reason="VLLM_SERVER_API_KEY or VLLM_SERVER_API_BASE or VLLM_SERVER_MODEL_NAME is not set",
)
def test_vllm_server_structured_output_regex_datetime(llm, datetime_obj):
    response: str = llm.structured_output(
        model=_model_name,
        constraint=RegexPattern.DATE_TIME_ISO_8601,
        messages=[
            Message.from_text(
                text="You are a helpful assistant.",
                role=Role.SYSTEM,
            ),
            Message.from_text(
                text=f"""
Today is {datetime_obj.strftime('%Y-%m-%d')}.
It is {datetime_obj.strftime('%H:%M:%S')} now.
I am in Beijing. Please tell me the time in Beijing in ISO 8601 format.
""",
                role=Role.USER,
            ),
        ],
    )
    printer.print("\n" + response, color='purple')
    datetime_obj_new = datetime.datetime.fromisoformat(response)
    assert datetime_obj_new.strftime('%Y-%m-%d %H:%M:%S') == datetime_obj.strftime('%Y-%m-%d %H:%M:%S')

@pytest.mark.skipif(
    (_api_key is None) or (_api_base is None) or (_model_name is None),
    reason="VLLM_SERVER_API_KEY or VLLM_SERVER_API_BASE or VLLM_SERVER_MODEL_NAME is not set",
)
def test_vllm_server_structured_output_choice(llm):
    choices = ["apple", "banana", "cherry", "kiwi", "orange", "pear", "pineapple", "strawberry", "watermelon"]
    response: str = llm.structured_output(
        model=_model_name,
        constraint=Choice(choices=choices),
        messages=[
            Message.from_text(
                text="You are a helpful assistant.",
                role=Role.SYSTEM,
            ),
            Message.from_text(
                text=f"Pick one fruit that is yellow from the following list: {', '.join(choices)}.",
                role=Role.USER,
            ),
        ],
    )
    printer.print("\n" + response, color='purple')
    assert response in choices

@pytest.mark.skipif(
    (_api_key is None) or (_api_base is None) or (_model_name is None),
    reason="VLLM_SERVER_API_KEY or VLLM_SERVER_API_BASE or VLLM_SERVER_MODEL_NAME is not set",
)
def test_vllm_server_select_tool(llm, date, tools, weather_messages):
    options = ["auto", "required", "none", {"type": "function", "function": {"name": "get_weather"}}]
    for option in options:
        printer.print(f"\nTool choice: [{option}]")
        response: Tuple[List[ToolCall], Optional[str]] = llm.select_tool(
            model=_model_name,
            tools=tools,
            tool_choice=option,
            messages=weather_messages,
        )
        handle_response(date, response)

@pytest.mark.skipif(
    (_api_key is None) or (_api_base is None) or (_model_name is None),
    reason="VLLM_SERVER_API_KEY or VLLM_SERVER_API_BASE or VLLM_SERVER_MODEL_NAME is not set",
)
@pytest.mark.asyncio
async def test_vllm_server_aselect_tool(llm, date, tools, weather_messages):
    options = ["auto", "required", "none", {"type": "function", "function": {"name": "get_weather"}}]
    for option in options:
        printer.print(f"\n[tool_choice: {option}]")
        response: Tuple[List[ToolCall], Optional[str]] = llm.aselect_tool(
            model=_model_name,
            tools=tools,
            tool_choice=option,
            messages=weather_messages,
        )
        handle_response(date, await response)
