import pytest
import os
import json
import re
import datetime

from bridgic.core.intelligence.base_llm import *
from bridgic.core.intelligence.protocol import *
from bridgic.core.utils.console import printer
from bridgic.llms.openai.openai_llm import OpenAILlm

_api_key = os.environ.get("OPENAI_API_KEY")
_model_name = os.environ.get("OPENAI_MODEL_NAME")

@pytest.fixture
def llm():
    return OpenAILlm(
        api_key=_api_key,
    )

@pytest.fixture
def date():
    date = datetime.datetime.now().strftime('%Y-%m-%d')
    return date

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


@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_server_chat(llm):
    response = llm.chat(
        model=_model_name,
        messages=[Message.from_text(text="Hello, how are you?", role=Role.USER)],
    )
    printer.print(response)
    assert response.message.role == Role.AI
    assert response.message.content is not None

@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_server_stream(llm):
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
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
@pytest.mark.asyncio
async def test_openai_server_achat(llm):
    response = await llm.achat(
        model=_model_name,
        messages=[Message.from_text(text="Hello, how are you?", role=Role.USER)],
    )
    printer.print(response)
    assert response.message.role == Role.AI
    assert response.message.content is not None

@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
@pytest.mark.asyncio
async def test_openai_server_astream(llm):
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
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_server_structured_output_pydantic_model(llm):
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
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
@pytest.mark.asyncio
async def test_openai_server_astructured_output_pydantic_model(llm):
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
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_server_structured_output_json_schema(llm):
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
        constraint=JsonSchema(schema_dict=schema),
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
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
@pytest.mark.asyncio
async def test_openai_server_astructured_output_json_schema(llm):
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
        constraint=JsonSchema(schema_dict=schema),
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
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
@pytest.mark.asyncio
async def test_openai_server_astructured_output_json_schema(llm):
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
        constraint=JsonSchema(schema_dict=schema),
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
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_server_structured_output_regex(llm):
    pattern = r"^Emails:\n(- [a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)(\n(- [a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+))*$"
    with pytest.raises(ValueError, match=r"Invalid constraint: constraint_type=\'regex\' pattern=.*"):
        llm.structured_output(
            model=_model_name,
            constraint=Regex(pattern=pattern),
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

@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
@pytest.mark.asyncio
async def test_openai_server_astructured_output_regex(llm):
    pattern = r"^Emails:\n(- [a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)(\n(- [a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+))*$"
    with pytest.raises(ValueError, match=r"Invalid constraint: constraint_type=\'regex\' pattern=.*"):
        await llm.astructured_output(
            model=_model_name,
            constraint=Regex(pattern=pattern),
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
    

@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_server_structured_output_ebnf_grammar(llm):
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
    with pytest.raises(ValueError, match=r"Invalid constraint: constraint_type=\'ebnf_grammar\' syntax=.*"):
        llm.structured_output(
            model=_model_name,
            constraint=EbnfGrammar(syntax=ebnf_syntax),
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

@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
@pytest.mark.asyncio
async def test_openai_server_astructured_output_ebnf_grammar(llm):
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
    with pytest.raises(ValueError, match=r"Invalid constraint: constraint_type=\'ebnf_grammar\' syntax=.*"):
        await llm.astructured_output(
            model=_model_name,
            constraint=EbnfGrammar(syntax=ebnf_syntax),
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

@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_server_select_tool(llm, date, tools):
    response, _ = llm.select_tool(
        model=_model_name,
        tools=tools,
        messages=[
            Message.from_text(
                text="You are a helpful assistant. You are good at calling the provided tools to solve problems.",
                role=Role.SYSTEM,
            ),
            Message.from_text(
                text=f"Today is {date}, get the weather of Tokyo and today's sports news.",
                role=Role.USER,
            )
        ],
    )
    printer.print("Tool Calls:")
    for tool_call in response:
        printer.print(json.dumps(tool_call.model_dump()), color='purple')
        if tool_call.name == "get_weather":
            assert tool_call.arguments["city"] == "Tokyo"
        if tool_call.name == "get_news":
            assert tool_call.arguments["date"] == date
            assert len(tool_call.arguments["topic"]) > 0



@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_server_select_tool_response(llm, tools):
    tool_calls, response = llm.select_tool(
        model=_model_name,
        tools=tools,
        messages=[
            Message.from_text(
                text="You are a helpful assistant. You are skilled at using the provided tools to solve problems. If the tool does not match the question, you can directly answer the answer",
                role=Role.SYSTEM,
            ),
            Message.from_text(
                text="What is 4 * 4 equal to when calculating",
                role=Role.USER,
            )
        ],
    )
    printer.print("Tool Calls:")
    assert len(tool_calls) == 0
    assert len(response) > 0


@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_server_empty_tool(llm):
    tool_calls, response = llm.select_tool(
        model=_model_name,
        tools=[],
        messages=[
            Message.from_text(
                text="You are a helpful assistant. You are skilled at using the provided tools to solve problems. If the tool does not match the question, you can directly answer the answer",
                role=Role.SYSTEM,
            ),
            Message.from_text(
                text="What is 4 * 4 equal to when calculating",
                role=Role.USER,
            )
        ],
    )
    assert len(tool_calls) == 0
    assert len(response) > 0


@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
@pytest.mark.asyncio
async def test_openai_server_aselect_tool(llm, date, tools):
    response, _ = await llm.aselect_tool(
        model=_model_name,
        tools=tools,
        messages=[
            Message.from_text(
                text="You are a helpful assistant. You are good at calling the provided tools to solve problems.",
                role=Role.SYSTEM,
            ),
            Message.from_text(
                text=f"Today is {date}, get the weather of Tokyo and today's sports news.",
                role=Role.USER,
            )
        ],
    )
    printer.print("Tool Calls:")
    for tool_call in response:
        printer.print(json.dumps(tool_call.model_dump()), color='purple')
        if tool_call.name == "get_weather":
            assert tool_call.arguments["city"] == "Tokyo"
        if tool_call.name == "get_news":
            assert tool_call.arguments["date"] == date
            assert len(tool_call.arguments["topic"]) > 0


@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_server_extras_name_user(llm: OpenAILlm):
    response = llm.chat(
        model=_model_name,
        messages=[
            Message.from_text(
                text="You are a helpful assistant.A friendly answer requires including the other party's name.",
                role=Role.SYSTEM,
            ),
            Message.from_text(
                text="Hey! What's your favorite food?",
                role=Role.USER,
                extras={"name": "Bob"},
            )
        ],
    )
    assert 'Bob' in response.message.content


@pytest.mark.skipif((_api_key is None) or (_model_name is None), reason="env not set")
def test_openai_server_extras_name_system(llm: OpenAILlm):
    response = llm.chat(
        model=_model_name,
        messages=[
            Message.from_text(
                text="Next, we will engage in role-playing. I will provide you with a name field, and you will need to reply with your name when I ask who you are?",
                role=Role.SYSTEM,
                extras={"name": "BtskBot"},
            ),
            Message.from_text(text="What is your name?", role=Role.USER),
        ],
    )
    assert "BtskBot" in response.message.content


@pytest.mark.skipif((_api_key is None) or (_model_name is None), reason="env not set")
def test_openai_server_extras_name_assistant(llm: OpenAILlm):
    response = llm.chat(
        model=_model_name,
        messages=[
            Message.from_text(text="Next, we will engage in role-playing. I will provide you with a name field, and you will need to reply with your name when I ask who you are?", role=Role.SYSTEM),
            Message.from_text(text="Hello!", role=Role.USER),
            Message.from_text(
                text="Hi there, I'm a helpful LLM!",
                role=Role.AI,
                extras={"name": "BtskBot"},
            ),
            Message.from_text(text="What's your name again?", role=Role.USER),
        ],
    )
    assert "BtskBot" in response.message.content


@pytest.mark.skipif((_api_key is None) or (_model_name is None), reason="env not set")
def test_openai_server_multiple_users(llm: OpenAILlm):
    response = llm.chat(
        model=_model_name,
        messages=[
            Message.from_text("You are a multi-user assistant.", role=Role.SYSTEM),
            Message.from_text("Hello!", role=Role.USER, extras={"name": "Alice"}),
            Message.from_text("Hi Alice! How are you?", role=Role.AI),
            Message.from_text("I'm good, thanks!", role=Role.USER, extras={"name": "Bob"}),
        ],
    )
    assert any(name in response.message.content for name in ["Alice", "Bob"])


@pytest.mark.skipif((_api_key is None) or (_model_name is None), reason="env not set")
def test_openai_server_no_name_extras(llm: OpenAILlm):
    response = llm.chat(
        model=_model_name,
        messages=[
            Message.from_text("You are a polite bot.", role=Role.SYSTEM),
            Message.from_text("How are you?", role=Role.USER, extras={}),
        ],
    )
    assert isinstance(response.message.content, str)
    assert len(response.message.content) > 0


@pytest.mark.skipif((_api_key is None) or (_model_name is None), reason="env not set")
def test_openai_server_invalid_name_type(llm: OpenAILlm):
    response = llm.chat(
        model=_model_name,
        messages=[
            Message.from_text("You are a robust bot.", role=Role.SYSTEM),
            Message.from_text("What do you think?", role=Role.USER, extras={"name": None}),
        ],
    )
    assert len(response.message.content)