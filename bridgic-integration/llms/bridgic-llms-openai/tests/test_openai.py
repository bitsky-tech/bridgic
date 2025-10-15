import pytest
import os
import json
import re
import datetime

from bridgic.core.model.base_llm import *
from bridgic.core.model.protocol import *
from bridgic.core.utils.console import printer
from bridgic.llms.openai.openai_llm import OpenAIConfiguration, OpenAILlm


_api_key = os.environ.get("OPENAI_API_KEY")
_model_name = os.environ.get("OPENAI_MODEL_NAME")

@pytest.fixture
def llm():
    return OpenAILlm(
        api_key=_api_key,
    )

def get_configuration_llm(configuration: OpenAIConfiguration):
    return OpenAILlm(
        api_key=_api_key,
        configuration=configuration
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
        thought: str = Field(description="The thought about the problem.", max_length=200)
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
        thought: str = Field(description="The thought about the problem.", max_length=200)
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
        constraint=JsonSchema(schema_dict=schema, name="ThinkAndAnswer"),
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
        constraint=JsonSchema(schema_dict=schema, name="ThinkAndAnswer"),
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
        constraint=JsonSchema(schema_dict=schema, name="ThinkAndAnswer"),
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
def test_openai_server_structured_output_regex_not_supported(llm: OpenAILlm):
    """Test that Regex constraints are not supported and raise ValueError."""
    pattern = r"^(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+(?P<year>\d{4})\s+at\s+(?P<hour>0?[1-9]|1[0-2])(?P<ampm>AM|PM)$"
    
    with pytest.raises(ValueError, match=r"Unsupported constraint type.*More info about OpenAI structured output"):
        llm.structured_output(
            model=_model_name,
            constraint=Regex(name="timestamp", pattern=pattern, description="Saves a timestamp in date + time in 24-hr format."),
            messages=[
                Message.from_text(
                    text="Use the timestamp tool to save a timestamp for August 7th 2025 at 10AM.",
                    role=Role.USER,
                ),
            ],
        )

@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
@pytest.mark.asyncio
async def test_openai_server_astructured_output_regex_not_supported(llm: OpenAILlm):
    """Test that Regex constraints are not supported in async version and raise ValueError."""
    pattern = "^(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+(?P<year>\d{4})\s+at\s+(?P<hour>0?[1-9]|1[0-2])(?P<ampm>AM|PM)$"
    
    with pytest.raises(ValueError, match=r"Unsupported constraint type.*More info about OpenAI structured output"):
        await llm.astructured_output(
            model=_model_name,
            constraint=Regex(name="timestamp", pattern=pattern, description="Saves a timestamp in date + time in 24-hr format."),
            messages=[
                Message.from_text(
                    text="Use the timestamp tool to save a timestamp for August 7th 2025 at 10AM.",
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
    with pytest.raises(ValueError, match=r"Unsupported constraint type.*More info about OpenAI structured output"):
        llm.structured_output(
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
    with pytest.raises(ValueError, match=r"Unsupported constraint type.*More info about OpenAI structured output"):
        await llm.astructured_output(
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

@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_server_structured_output_lark_grammar_not_supported(llm):
    """Test that LarkGrammar constraints are not supported and raise ValueError."""
    lark_syntax = """start: expr
expr: term (SP ADD SP term)* -> add
| term
term: factor (SP MUL SP factor)* -> mul
| factor
factor: INT
SP: " "
ADD: "+"
MUL: "*"
%import common.INT"""

    with pytest.raises(ValueError, match=r"Unsupported constraint type.*More info about OpenAI structured output"):
        llm.structured_output(
            model=_model_name,
            temperature=0,
            constraint=LarkGrammar(name="math_exp", syntax=lark_syntax, description="Creates valid mathematical expressions"),
            messages=[
                Message.from_text(
                    text="Use the math_exp tool to add four plus four.",
                    role=Role.USER,
                ),
            ],
        )


@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
@pytest.mark.asyncio
async def test_openai_server_astructured_output_lark_grammar_not_supported(llm):
    """Test that LarkGrammar constraints are not supported in async version and raise ValueError."""
    lark_syntax = """start: expr
expr: term (SP ADD SP term)* -> add
| term
term: factor (SP MUL SP factor)* -> mul
| factor
factor: INT
SP: " "
ADD: "+"
MUL: "*"
%import common.INT"""

    with pytest.raises(ValueError, match=r"Unsupported constraint type.*More info about OpenAI structured output"):
        await llm.astructured_output(
            model=_model_name,
            temperature=0,
            constraint=LarkGrammar(name="math_exp", syntax=lark_syntax, description="Creates valid mathematical expressions"),
            messages=[
                Message.from_text(
                    text="Use the math_exp tool to add four plus four.",
                    role=Role.USER,
                ),
            ],
        )


@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_server_structured_output_choice_not_supported(llm):
    """Test that Choice constraints are not supported and raise ValueError."""
    with pytest.raises(ValueError, match=r"Unsupported constraint type.*More info about OpenAI structured output"):
        llm.structured_output(
            model=_model_name,
            constraint=Choice(choices=["option1", "option2", "option3"]),
            messages=[
                Message.from_text(
                    text="Choose one of the options.",
                    role=Role.USER,
                ),
            ],
        )


@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
@pytest.mark.asyncio
async def test_openai_server_astructured_output_choice_not_supported(llm):
    """Test that Choice constraints are not supported in async version and raise ValueError."""
    with pytest.raises(ValueError, match=r"Unsupported constraint type.*More info about OpenAI structured output"):
        await llm.astructured_output(
            model=_model_name,
            constraint=Choice(choices=["option1", "option2", "option3"]),
            messages=[
                Message.from_text(
                    text="Choose one of the options.",
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


@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_server_function_call_with_blocks(llm: OpenAILlm):
    """
    Test function calling using ToolCallBlock and ToolResultBlock.
    This test demonstrates how to simulate a complete function call workflow.
    """
    # Create messages simulating a conversation with function calls
    messages = [
        Message.from_text(
            text="You are a helpful assistant that can call functions to get information.",
            role=Role.SYSTEM
        ),
        Message.from_text(
            text="What's the weather like in Tokyo?",
            role=Role.USER
        ),
        # AI response with tool call using the new from_tool_call method
        Message(
            role=Role.AI,
            blocks=[
                TextBlock(text="I'll check the weather for you."), 
                ToolCallBlock(id="call_123", name="get_weather", arguments={"city": "Tokyo", "unit": "celsius"})
            ]
        ),
        # Tool result message using the new from_tool_result method
        Message.from_tool_result(
            tool_id="call_123",
            content="The weather in Tokyo is 22°C and sunny."
        ),
        # Follow-up user message
        Message.from_text(
            text="Thanks! What about the weather in Paris?",
            role=Role.USER
        )
    ]
    
    # Test that we can create and access the blocks correctly
    assert len(messages) == 5
    
    # Verify the tool call block
    ai_message = messages[2]
    assert ai_message.role == Role.AI
    assert len(ai_message.blocks) == 2
    assert isinstance(ai_message.blocks[0], TextBlock)
    assert isinstance(ai_message.blocks[1], ToolCallBlock)
    
    tool_call = ai_message.blocks[1]
    assert tool_call.id == "call_123"
    assert tool_call.name == "get_weather"
    assert tool_call.arguments["city"] == "Tokyo"
    assert tool_call.arguments["unit"] == "celsius"
    
    # Verify the tool result block
    tool_message = messages[3]
    assert tool_message.role == Role.TOOL
    assert len(tool_message.blocks) == 1
    assert isinstance(tool_message.blocks[0], ToolResultBlock)
    
    tool_result = tool_message.blocks[0]
    assert tool_result.id == "call_123"
    assert "Tokyo" in tool_result.content
    assert "22°C" in tool_result.content
    
    # Test that we can extract text content from mixed blocks
    ai_text_content = ai_message.content
    assert "I'll check the weather for you." in ai_text_content
    
    printer.print("Function call workflow test passed!")
    printer.print(f"Tool call: {tool_call.model_dump()}")
    printer.print(f"Tool result: {tool_result.model_dump()}")


@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_server_multiple_function_calls_with_blocks(llm: OpenAILlm):
    """
    Test multiple function calls using ToolCallBlock and ToolResultBlock.
    This test demonstrates handling multiple parallel function calls.
    """
    # Create conversation with multiple function calls
    messages = [
        Message.from_text(
            text="You are a helpful assistant that can call multiple functions simultaneously.",
            role=Role.SYSTEM
        ),
        Message.from_text(
            text="Get me the weather in Tokyo and today's tech news.",
            role=Role.USER
        ),
        # AI response with multiple tool calls
        Message(
            role=Role.AI,
            blocks=[
                TextBlock(text="I'll get both pieces of information for you."),
                ToolCallBlock(id="call_weather_123", name="get_weather", arguments={"city": "Tokyo", "unit": "celsius"}),
                ToolCallBlock(id="call_news_456", name="get_news", arguments={"topic": "technology", "date": "2024-01-15"})
            ]
        ),
        # Tool results using the new from_tool_result method
        Message.from_tool_result(
            tool_id="call_weather_123",
            content="Tokyo weather: 22°C, sunny with light clouds."
        ),
        Message.from_tool_result(
            tool_id="call_news_456",
            content="Latest tech news: AI breakthrough in quantum computing announced."
        )
    ]
    
    # Verify the structure
    assert len(messages) == 5
    
    # Check AI message with multiple tool calls
    ai_message = messages[2]
    assert ai_message.role == Role.AI
    assert len(ai_message.blocks) == 3  # 1 text + 2 tool calls
    
    # Verify tool calls
    tool_calls = [block for block in ai_message.blocks if isinstance(block, ToolCallBlock)]
    assert len(tool_calls) == 2
    
    weather_tool_call = tool_calls[0]
    news_tool_call = tool_calls[1]
    
    assert weather_tool_call.name == "get_weather"
    assert news_tool_call.name == "get_news"
    
    # Verify tool results
    weather_result_msg = messages[3]
    news_result_msg = messages[4]
    
    assert weather_result_msg.role == Role.TOOL
    assert news_result_msg.role == Role.TOOL
    
    weather_result_block = weather_result_msg.blocks[0]
    news_result_block = news_result_msg.blocks[0]
    
    assert weather_result_block.id == "call_weather_123"
    assert news_result_block.id == "call_news_456"
    
    printer.print("Multiple function calls test passed!")
    printer.print(f"Weather call: {weather_tool_call.model_dump()}")
    printer.print(f"News call: {news_tool_call.model_dump()}")
    printer.print(f"Weather result: {weather_result_block.model_dump()}")
    printer.print(f"News result: {news_result_block.model_dump()}")


@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_server_function_call_error_handling(llm: OpenAILlm):
    """
    Test function call error handling using ToolResultBlock.
    This test demonstrates how to handle function call errors.
    """
    # Create conversation with error handling
    messages = [
        Message.from_text(
            text="You are a helpful assistant. If a function call fails, explain the error to the user.",
            role=Role.SYSTEM
        ),
        Message.from_text(
            text="Get the weather for InvalidCity.",
            role=Role.USER
        ),
        # AI response with tool call
        Message(
            role=Role.AI,
            blocks=[
                TextBlock(text="I'll check the weather for that city."),
                ToolCallBlock(id="call_error_789", name="get_weather", arguments={"city": "InvalidCity", "unit": "celsius"})
            ]
        ),
        # Tool error result using the new from_tool_result method
        Message.from_tool_result(
            tool_id="call_error_789",
            content="Error: City 'InvalidCity' not found. Please provide a valid city name."
        )
    ]
    
    # Verify error handling structure
    assert len(messages) == 4
    
    # Check the error result
    error_message = messages[3]
    assert error_message.role == Role.TOOL
    error_block = error_message.blocks[0]
    assert isinstance(error_block, ToolResultBlock)
    assert error_block.id == "call_error_789"
    assert "Error:" in error_block.content
    assert "not found" in error_block.content
    
    printer.print("Function call error handling test passed!")
    printer.print(f"Error result: {error_block.model_dump()}")

@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_server_message_conversion(llm: OpenAILlm):
    """
    Test the message conversion functionality for tool calls and tool results.
    This test verifies that Bridgic messages are correctly converted to OpenAI format.
    """
    # Test conversion of tool call message
    tool_call_message = Message(
        role=Role.AI,
        blocks=[
            TextBlock(text="I'll check the weather for you."),
            ToolCallBlock(id="call_weather_123", name="get_weather", arguments={"city": "Tokyo", "unit": "celsius"})
        ]
    )
    
    # Test conversion of tool result message
    tool_result_message = Message.from_tool_result(
        tool_id="call_weather_123",
        content="The weather in Tokyo is 22°C and sunny."
    )
    
    # Test conversion of regular text message
    text_message = Message.from_text("Hello, how are you?", role=Role.USER)
    
    # Convert messages to OpenAI format
    try:
        converted_tool_call = llm._convert_chat_completions_message(tool_call_message)
        converted_tool_result = llm._convert_chat_completions_message(tool_result_message)
        converted_text = llm._convert_chat_completions_message(text_message)
        
        # Verify conversions
        assert converted_tool_call["role"] == "assistant"
        assert "tool_calls" in converted_tool_call
        assert len(converted_tool_call["tool_calls"]) == 1
        assert converted_tool_call["tool_calls"][0]["function"]["name"] == "get_weather"
        
        assert converted_tool_result["role"] == "tool"
        assert converted_tool_result["tool_call_id"] == "call_weather_123"
        assert "The weather in Tokyo is 22°C and sunny." in converted_tool_result["content"]
        
        assert converted_text["role"] == "user"
        assert converted_text["content"] == "Hello, how are you?"
        
        printer.print("Message conversion test passed!")
        printer.print(f"Tool call conversion: {converted_tool_call}")
        printer.print(f"Tool result conversion: {converted_tool_result}")
        printer.print(f"Text message conversion: {converted_text}")
        
    except Exception as e:
        printer.print(f"Conversion error: {e}")
        raise


@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_server_tool_call_edge_cases(llm: OpenAILlm):
    """
    Test edge cases for tool call functionality.
    This test covers various edge cases and error scenarios.
    """
    # Test 1: Empty tool call arguments
    empty_args_message = Message(
        role=Role.AI,
        blocks=[
            TextBlock(text="Calling a function with no arguments."),
            ToolCallBlock(id="call_empty", name="get_time", arguments={})
        ]
    )
    
    converted_empty = llm._convert_chat_completions_message(empty_args_message)
    assert converted_empty["tool_calls"][0]["function"]["arguments"] == "{}"
    
    # Test 2: Tool call with complex nested arguments
    complex_args = {
        "user": {"name": "John", "age": 30},
        "preferences": ["weather", "news"],
        "settings": {"unit": "celsius", "language": "en"}
    }
    complex_message = Message(
        role=Role.AI,
        blocks=[
            ToolCallBlock(id="call_complex", name="get_user_data", arguments=complex_args)
        ]
    )
    
    converted_complex = llm._convert_chat_completions_message(complex_message)
    parsed_args = json.loads(converted_complex["tool_calls"][0]["function"]["arguments"])
    assert parsed_args["user"]["name"] == "John"
    assert parsed_args["preferences"] == ["weather", "news"]
    
    # Test 3: Tool result with empty content
    empty_result = Message.from_tool_result(
        tool_id="call_empty_result",
        content=""
    )
    
    converted_empty_result = llm._convert_chat_completions_message(empty_result)
    assert converted_empty_result["content"] == ""
    assert converted_empty_result["tool_call_id"] == "call_empty_result"
    
    # Test 4: Tool result with special characters
    special_content = "Result with special chars: <>&\"'\\n\\t"
    special_result = Message.from_tool_result(
        tool_id="call_special",
        content=special_content
    )
    
    converted_special = llm._convert_chat_completions_message(special_result)
    assert converted_special["content"] == special_content
    
    printer.print("Tool call edge cases test passed!")
    printer.print(f"Empty args: {converted_empty}")
    printer.print(f"Complex args: {converted_complex}")
    printer.print(f"Empty result: {converted_empty_result}")
    printer.print(f"Special content: {converted_special}")


@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_server_multiple_tool_calls_conversion(llm: OpenAILlm):
    """
    Test conversion of messages with multiple tool calls.
    This test verifies handling of parallel tool calls.
    """
    # Create a message with multiple tool calls
    multi_tool_message = Message(
        role=Role.AI,
        blocks=[
            TextBlock(text="I'll get both weather and news for you."),
            ToolCallBlock(id="call_weather_1", name="get_weather", arguments={"city": "Tokyo"}),
            ToolCallBlock(id="call_news_2", name="get_news", arguments={"topic": "technology"}),
            ToolCallBlock(id="call_time_3", name="get_time", arguments={"timezone": "UTC"})
        ]
    )
    
    # Convert the message
    converted = llm._convert_chat_completions_message(multi_tool_message)
    
    # Verify the conversion
    assert converted["role"] == "assistant"
    assert "I'll get both weather and news for you." in converted["content"]
    assert len(converted["tool_calls"]) == 3
    
    # Verify each tool call
    tool_names = [tc["function"]["name"] for tc in converted["tool_calls"]]
    assert "get_weather" in tool_names
    assert "get_news" in tool_names
    assert "get_time" in tool_names
    
    # Verify tool call IDs
    tool_ids = [tc["id"] for tc in converted["tool_calls"]]
    assert "call_weather_1" in tool_ids
    assert "call_news_2" in tool_ids
    assert "call_time_3" in tool_ids
    
    printer.print("Multiple tool calls conversion test passed!")
    printer.print(f"Converted message: {converted}")


@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_server_tool_call_error_handling(llm: OpenAILlm):
    """
    Test error handling for tool call conversions.
    This test verifies proper error handling for invalid inputs.
    """
    # Test 1: Tool message without ToolResultBlock
    invalid_tool_message = Message(
        role=Role.TOOL,
        blocks=[TextBlock(text="This should have a ToolResultBlock")]
    )
    
    with pytest.raises(ValueError, match="Tool message must contain a ToolResultBlock with an ID"):
        llm._convert_chat_completions_message(invalid_tool_message)
    
    # Test 2: Tool message with multiple ToolResultBlocks (should use the first one)
    multi_result_message = Message(
        role=Role.TOOL,
        blocks=[
            ToolResultBlock(id="call_1", content="First result"),
            ToolResultBlock(id="call_2", content="Second result")
        ]
    )
    
    converted = llm._convert_chat_completions_message(multi_result_message)
    assert converted["tool_call_id"] == "call_1"  # Should use the first one
    
    # Test 3: Invalid JSON in tool call arguments (should be handled gracefully)
    try:
        # This should not raise an exception during conversion
        # The JSON error would be caught when OpenAI tries to parse it
        invalid_json_message = Message(
            role=Role.AI,
            blocks=[
                ToolCallBlock(id="call_invalid", name="test", arguments={"invalid": "json"})
            ]
        )
        converted_invalid = llm._convert_chat_completions_message(invalid_json_message)
        assert converted_invalid["tool_calls"][0]["function"]["name"] == "test"
    except Exception as e:
        # If it does raise an exception, it should be a JSON-related one
        assert "json" in str(e).lower() or "serialize" in str(e).lower()
    
    printer.print("Tool call error handling test passed!")


@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_server_tool_call_with_extras(llm: OpenAILlm):
    """
    Test tool call messages with extras metadata.
    This test verifies that extras are properly passed through.
    """
    # Test tool call with extras
    tool_call_with_extras = Message(
        role=Role.AI,
        blocks=[
            TextBlock(text="Calling with metadata."),
            ToolCallBlock(id="call_meta", name="get_data", arguments={"param": "value"})
        ],
        extras={"priority": "high", "source": "user", "timestamp": "2024-01-15"}
    )
    
    converted = llm._convert_chat_completions_message(tool_call_with_extras)
    
    # Verify extras are included
    assert converted["priority"] == "high"
    assert converted["source"] == "user"
    assert converted["timestamp"] == "2024-01-15"
    assert converted["role"] == "assistant"
    assert len(converted["tool_calls"]) == 1
    
    # Test tool result with extras
    tool_result_with_extras = Message.from_tool_result(
        tool_id="call_meta",
        content="Result with metadata",
        extras={"execution_time": "0.5s", "status": "success"}
    )
    
    converted_result = llm._convert_chat_completions_message(tool_result_with_extras)
    
    # Verify extras are included
    assert converted_result["execution_time"] == "0.5s"
    assert converted_result["status"] == "success"
    assert converted_result["role"] == "tool"
    assert converted_result["tool_call_id"] == "call_meta"
    
    printer.print("Tool call with extras test passed!")
    printer.print(f"Tool call with extras: {converted}")
    printer.print(f"Tool result with extras: {converted_result}")



@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_server_select_tool_tool_call_roundtrip_conversion(llm: OpenAILlm, tools):
    """
    Test roundtrip conversion of tool call messages.
    This test simulates a complete conversation with tool calls.
    """
    def get_weather(city: str):
        return f"{city} weather: 22°C, sunny with light clouds."
    def get_news(topic: str, date: str):
        return f"Latest tech news: {topic} on {date}."
    # Simulate a complete conversation
    conversation = [
        # System message
        Message.from_text("You are a helpful assistant.", role=Role.SYSTEM),
        
        # User message
        Message.from_text("Get me the weather in Tokyo and news about technology at 2024-01-15.", role=Role.USER),
    ]
    
    response,msg = llm.select_tool(conversation, model=_model_name, tools=tools)
    print(response)
    assert len(response) == 2
    assert response[0].name == "get_weather"
    assert response[1].name == "get_news"
    assert response[0].arguments["city"] == "Tokyo"
    assert response[1].arguments["topic"] == "technology"
    assert response[1].arguments["date"] == "2024-01-15"
    # add ai message with select_tool result to conversation
    conversation.append(Message.from_tool_call(tool_calls=response, text=msg))
    # add tool result to conversation
    for tool_call in response:
        if tool_call.name == "get_weather":
            result = get_weather(tool_call.arguments["city"])
            tool_result = Message.from_tool_result(
                tool_id=tool_call.id,
                content=result
            )
            conversation.append(tool_result)
        if tool_call.name == "get_news":
            result = get_news(tool_call.arguments["topic"], tool_call.arguments["date"])
            tool_result = Message.from_tool_result(
                tool_id=tool_call.id,
                content=result
            )
            conversation.append(tool_result)
    conversation.append(Message.from_text("Thanks! What about the weather in Paris?", role=Role.USER))
    response,_ = llm.select_tool(conversation, model=_model_name, tools=tools)
    print(response)
    assert len(response) == 1
    assert response[0].name == "get_weather"
    assert response[0].arguments["city"] == "Paris"


@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
@pytest.mark.asyncio
async def test_openai_server_aselect_tool_tool_call_roundtrip_conversion(llm: OpenAILlm, tools):
    """
    Test roundtrip conversion of tool call messages.
    This test simulates a complete conversation with tool calls.
    """
    def get_weather(city: str):
        return f"{city} weather: 22°C, sunny with light clouds."
    def get_news(topic: str, date: str):
        return f"Latest tech news: {topic} on {date}."
    # Simulate a complete conversation
    conversation = [
        # System message
        Message.from_text("You are a helpful assistant.", role=Role.SYSTEM),
        
        # User message
        Message.from_text("Get me the weather in Tokyo and news about technology at 2024-01-15.", role=Role.USER),
    ]
    
    response,msg = await llm.aselect_tool(conversation, model=_model_name, tools=tools)
    print(response)
    assert len(response) == 2
    assert response[0].name == "get_weather"
    assert response[1].name == "get_news"
    assert response[0].arguments["city"] == "Tokyo"
    assert response[1].arguments["topic"] == "technology"
    assert response[1].arguments["date"] == "2024-01-15"
    # add ai message with select_tool result to conversation
    conversation.append(Message.from_tool_call(tool_calls=response, text=msg))
    # add tool result to conversation
    for tool_call in response:
        if tool_call.name == "get_weather":
            result = get_weather(tool_call.arguments["city"])
            tool_result = Message.from_tool_result(
                tool_id=tool_call.id,
                content=result
            )
            conversation.append(tool_result)
        if tool_call.name == "get_news":
            result = get_news(tool_call.arguments["topic"], tool_call.arguments["date"])
            tool_result = Message.from_tool_result(
                tool_id=tool_call.id,
                content=result
            )
            conversation.append(tool_result)
    conversation.append(Message.from_text("Thanks! What about the weather in Paris?", role=Role.USER))
    response,_ = await llm.aselect_tool(conversation, model=_model_name, tools=tools)
    print(response)
    assert len(response) == 1
    assert response[0].name == "get_weather"
    assert response[0].arguments["city"] == "Paris"


@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_server_tool_call_roundtrip_conversion(llm: OpenAILlm):
    """
    Test roundtrip conversion of tool call messages.
    This test simulates a complete conversation with tool calls.
    """
    # Simulate a complete conversation
    conversation = [
        # System message
        Message.from_text("You are a helpful assistant.", role=Role.SYSTEM),
        
        # User message
        Message.from_text("Get me the weather in Tokyo and today's news.", role=Role.USER),
        
        # AI response with multiple tool calls
        Message(
            role=Role.AI,
            blocks=[
                TextBlock(text="I'll get both pieces of information for you."),
                ToolCallBlock(id="call_weather_001", name="get_weather", arguments={"city": "Tokyo", "unit": "celsius"}),
                ToolCallBlock(id="call_news_002", name="get_news", arguments={"topic": "technology", "date": "2024-01-15"})
            ]
        ),
        
        # Tool results
        Message.from_tool_result(
            tool_id="call_weather_001",
            content="Tokyo weather: 22°C, sunny with light clouds."
        ),
        Message.from_tool_result(
            tool_id="call_news_002",
            content="Latest tech news: AI breakthrough in quantum computing announced."
        ),
        
        # Follow-up user message
        Message.from_text("Thanks! What about the weather in Paris?", role=Role.USER),
        
        # Another AI response with tool call
        Message(
            role=Role.AI,
            blocks=[
                TextBlock(text="I'll check the weather in Paris for you."),
                ToolCallBlock(id="call_weather_003", name="get_weather", arguments={"city": "Paris", "unit": "celsius"})
            ]
        ),
        
        # Final tool result
        Message.from_tool_result(
            tool_id="call_weather_003",
            content="Paris weather: 15°C, partly cloudy."
        )
    ]
    
    # Convert all messages
    converted_messages = []
    for i, message in enumerate(conversation):
        try:
            converted = llm._convert_chat_completions_message(message)
            converted_messages.append(converted)
            
            # Verify basic structure
            assert "role" in converted
            assert converted["role"] in ["system", "user", "assistant", "tool"]
            
            # Verify role-specific content
            if message.role == Role.AI:
                if any(isinstance(block, ToolCallBlock) for block in message.blocks):
                    assert "tool_calls" in converted
                    assert len(converted["tool_calls"]) > 0
                else:
                    assert "content" in converted
            elif message.role == Role.TOOL:
                assert "tool_call_id" in converted
                assert "content" in converted
            elif message.role in [Role.SYSTEM, Role.USER]:
                assert "content" in converted
                
        except Exception as e:
            printer.print(f"Error converting message {i}: {e}")
            raise
    
    # Verify we converted all messages
    assert len(converted_messages) == len(conversation)
    
    # Verify specific conversions
    assert converted_messages[0]["role"] == "system"
    assert converted_messages[1]["role"] == "user"
    assert converted_messages[2]["role"] == "assistant"
    assert len(converted_messages[2]["tool_calls"]) == 2
    assert converted_messages[3]["role"] == "tool"
    assert converted_messages[3]["tool_call_id"] == "call_weather_001"
    assert converted_messages[4]["role"] == "tool"
    assert converted_messages[4]["tool_call_id"] == "call_news_002"
    
    printer.print("Roundtrip conversion test passed!")
    printer.print(f"Converted {len(converted_messages)} messages successfully")
    
    # Print summary
    for i, (original, converted) in enumerate(zip(conversation, converted_messages)):
        printer.print(f"Message {i}: {original.role.value} -> {converted['role']}")
        if original.role == Role.AI and any(isinstance(block, ToolCallBlock) for block in original.blocks):
            printer.print(f"  - Tool calls: {len(converted.get('tool_calls', []))}")
        elif original.role == Role.TOOL:
            printer.print(f"  - Tool call ID: {converted.get('tool_call_id', 'None')}")


@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_server_tool_call_performance(llm: OpenAILlm):
    """
    Test performance of tool call conversions with large numbers of messages.
    This test verifies that the conversion process is efficient.
    """
    import time
    
    # Create a large number of tool call messages
    messages = []
    for i in range(100):
        message = Message(
            role=Role.AI,
            blocks=[
                TextBlock(text=f"Calling function {i}"),
                ToolCallBlock(
                    id=f"call_{i}",
                    name="test_function",
                    arguments={"index": i, "data": f"test_data_{i}"}
                )
            ]
        )
        messages.append(message)
    
    # Measure conversion time
    start_time = time.time()
    converted_messages = []
    
    for message in messages:
        converted = llm._convert_chat_completions_message(message)
        converted_messages.append(converted)
    
    end_time = time.time()
    conversion_time = end_time - start_time
    
    # Verify all messages were converted
    assert len(converted_messages) == 100
    
    # Verify conversion quality
    for i, converted in enumerate(converted_messages):
        assert converted["role"] == "assistant"
        assert len(converted["tool_calls"]) == 1
        assert converted["tool_calls"][0]["id"] == f"call_{i}"
        assert converted["tool_calls"][0]["function"]["name"] == "test_function"
    
    printer.print(f"Performance test passed! Converted 100 messages in {conversion_time:.3f} seconds")
    printer.print(f"Average time per message: {conversion_time/100*1000:.2f} ms")


@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_server_tool_call_content_types(llm: OpenAILlm):
    """
    Test tool call messages with different content types and structures.
    This test verifies handling of various data types in tool arguments.
    """
    # Test with different data types in arguments
    test_cases = [
        # String arguments
        {"name": "string_test", "args": {"text": "hello world", "description": "test string"}},
        
        # Numeric arguments
        {"name": "numeric_test", "args": {"count": 42, "price": 99.99, "ratio": 0.85}},
        
        # Boolean arguments
        {"name": "boolean_test", "args": {"enabled": True, "disabled": False}},
        
        # List arguments
        {"name": "list_test", "args": {"items": ["apple", "banana", "cherry"], "numbers": [1, 2, 3]}},
        
        # Nested object arguments
        {"name": "object_test", "args": {
            "user": {"id": 123, "name": "John", "active": True},
            "settings": {"theme": "dark", "notifications": False}
        }},
        
        # Mixed arguments
        {"name": "mixed_test", "args": {
            "title": "Test Title",
            "count": 5,
            "enabled": True,
            "tags": ["test", "example"],
            "metadata": {"version": "1.0", "author": "Test User"}
        }},
        
        # Special characters and unicode
        {"name": "unicode_test", "args": {
            "text": "Hello 世界 🌍",
            "emoji": "🚀✨🎉",
            "special": "Special chars: <>&\"'\\n\\t"
        }}
    ]
    
    for i, test_case in enumerate(test_cases):
        message = Message(
            role=Role.AI,
            blocks=[
                ToolCallBlock(
                    id=f"call_type_{i}",
                    name=test_case["name"],
                    arguments=test_case["args"]
                )
            ]
        )
        
        # Convert the message
        converted = llm._convert_chat_completions_message(message)
        
        # Verify conversion
        assert converted["role"] == "assistant"
        assert len(converted["tool_calls"]) == 1
        
        tool_call = converted["tool_calls"][0]
        assert tool_call["id"] == f"call_type_{i}"
        assert tool_call["function"]["name"] == test_case["name"]
        
        # Parse and verify arguments
        parsed_args = json.loads(tool_call["function"]["arguments"])
        assert parsed_args == test_case["args"]
        
        printer.print(f"Content type test {i+1} passed: {test_case['name']}")
    
    printer.print("All content type tests passed!")


@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_server_tool_call_validation(llm: OpenAILlm):
    """
    Test validation of tool call message structure.
    This test verifies that invalid message structures are handled properly.
    """
    # Test 1: Message with only tool calls (no text)
    tool_only_message = Message(
        role=Role.AI,
        blocks=[
            ToolCallBlock(id="call_only", name="test_function", arguments={"param": "value"})
        ]
    )
    
    converted = llm._convert_chat_completions_message(tool_only_message)
    assert converted["role"] == "assistant"
    assert converted["content"] == ""  # Should be empty string when no text content
    assert len(converted["tool_calls"]) == 1
    
    # Test 2: Message with only text (no tool calls)
    text_only_message = Message(
        role=Role.AI,
        blocks=[TextBlock(text="Just text, no tool calls")]
    )
    
    converted_text = llm._convert_chat_completions_message(text_only_message)
    assert converted_text["role"] == "assistant"
    assert converted_text["content"] == "Just text, no tool calls"
    assert "tool_calls" not in converted_text
    
    # Test 3: Message with mixed content
    mixed_message = Message(
        role=Role.AI,
        blocks=[
            TextBlock(text="I'll help you with that."),
            ToolCallBlock(id="call_mixed", name="helper_function", arguments={"task": "help"}),
            TextBlock(text="Let me call the function now.")
        ]
    )
    
    converted_mixed = llm._convert_chat_completions_message(mixed_message)
    assert converted_mixed["role"] == "assistant"
    assert "I'll help you with that." in converted_mixed["content"]
    assert "Let me call the function now." in converted_mixed["content"]
    assert len(converted_mixed["tool_calls"]) == 1
    
    # Test 4: Tool result with empty ID
    empty_id_result = Message(
        role=Role.TOOL,
        blocks=[ToolResultBlock(id="", content="Empty ID result")]
    )
    
    converted_empty_id = llm._convert_chat_completions_message(empty_id_result)
    assert converted_empty_id["role"] == "tool"
    assert converted_empty_id["tool_call_id"] == ""
    assert converted_empty_id["content"] == "Empty ID result"
    
    printer.print("Tool call validation test passed!")
    printer.print(f"Tool only: {converted}")
    printer.print(f"Text only: {converted_text}")
    printer.print(f"Mixed content: {converted_mixed}")
    printer.print(f"Empty ID result: {converted_empty_id}")


@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_server_tool_call_integration_scenario(llm: OpenAILlm):
    """
    Test a complete integration scenario with tool calls.
    This test simulates a real-world usage scenario.
    """
    # Simulate a weather assistant scenario
    scenario_messages = [
        # System prompt
        Message.from_text(
            "You are a weather assistant. You can get weather information and news. "
            "Always provide helpful and accurate information.",
            role=Role.SYSTEM
        ),
        
        # User request
        Message.from_text(
            "I'm planning a trip to Tokyo next week. Can you get me the weather forecast "
            "and any relevant travel news?",
            role=Role.USER
        ),
        
        # AI response with multiple tool calls
        Message(
            role=Role.AI,
            blocks=[
                TextBlock(text="I'll help you plan your trip to Tokyo! Let me get the weather forecast and travel news for you."),
                ToolCallBlock(
                    id="weather_call_001",
                    name="get_weather_forecast",
                    arguments={
                        "city": "Tokyo",
                        "days": 7,
                        "unit": "celsius",
                        "include_hourly": True
                    }
                ),
                ToolCallBlock(
                    id="news_call_002",
                    name="get_travel_news",
                    arguments={
                        "destination": "Tokyo",
                        "category": "travel",
                        "limit": 5
                    }
                )
            ]
        ),
        
        # Weather tool result
        Message.from_tool_result(
            tool_id="weather_call_001",
            content="Tokyo 7-day forecast: Monday 22°C sunny, Tuesday 24°C partly cloudy, "
                   "Wednesday 20°C rainy, Thursday 23°C sunny, Friday 25°C clear, "
                   "Saturday 26°C sunny, Sunday 24°C partly cloudy. "
                   "Best days for outdoor activities: Monday, Thursday, Friday, Saturday."
        ),
        
        # News tool result
        Message.from_tool_result(
            tool_id="news_call_002",
            content="Recent Tokyo travel news: 1) Cherry blossom season starting early this year, "
                   "2) New direct flight routes from major cities, 3) Updated visa requirements, "
                   "4) Popular tourist attractions reopening, 5) Local transportation updates."
        ),
        
        # Follow-up user question
        Message.from_text(
            "Thanks! What about the weather in Kyoto? I'm also planning to visit there.",
            role=Role.USER
        ),
        
        # Another AI response
        Message(
            role=Role.AI,
            blocks=[
                TextBlock(text="I'll check the weather in Kyoto for you as well!"),
                ToolCallBlock(
                    id="kyoto_weather_003",
                    name="get_weather_forecast",
                    arguments={
                        "city": "Kyoto",
                        "days": 7,
                        "unit": "celsius",
                        "include_hourly": False
                    }
                )
            ]
        ),
        
        # Kyoto weather result
        Message.from_tool_result(
            tool_id="kyoto_weather_003",
            content="Kyoto 7-day forecast: Monday 20°C sunny, Tuesday 22°C partly cloudy, "
                   "Wednesday 18°C rainy, Thursday 21°C sunny, Friday 23°C clear, "
                   "Saturday 24°C sunny, Sunday 22°C partly cloudy. "
                   "Similar weather pattern to Tokyo, but slightly cooler."
        )
    ]
    
    # Convert all messages and verify
    converted_messages = []
    for i, message in enumerate(scenario_messages):
        converted = llm._convert_chat_completions_message(message)
        converted_messages.append(converted)
        
        # Verify basic structure
        assert "role" in converted
        assert converted["role"] in ["system", "user", "assistant", "tool"]
        
        # Verify content based on message type
        if message.role == Role.SYSTEM:
            assert "content" in converted
            assert "weather assistant" in converted["content"].lower()
        elif message.role == Role.USER:
            assert "content" in converted
            assert "tokyo" in converted["content"].lower() or "kyoto" in converted["content"].lower()
        elif message.role == Role.AI:
            if any(isinstance(block, ToolCallBlock) for block in message.blocks):
                assert "tool_calls" in converted
                assert len(converted["tool_calls"]) > 0
            else:
                assert "content" in converted
        elif message.role == Role.TOOL:
            assert "tool_call_id" in converted
            assert "content" in converted
            # Check for relevant keywords based on tool result content
            tool_content = converted["content"].lower()
            # Check for weather-related keywords OR travel/news-related keywords
            weather_keywords = ["weather", "forecast", "sunny", "cloudy", "rainy", "temperature", "°c", "°f"]
            travel_keywords = ["travel", "news", "flight", "visa", "tourist", "transportation", "cherry", "blossom"]
            assert any(keyword in tool_content for keyword in weather_keywords + travel_keywords)
    
    # Verify specific scenario elements
    assert len(converted_messages) == 8
    
    # Check tool calls
    ai_messages = [msg for msg in converted_messages if msg["role"] == "assistant"]
    assert len(ai_messages) == 2
    assert len(ai_messages[0]["tool_calls"]) == 2  # Weather + news
    assert len(ai_messages[1]["tool_calls"]) == 1  # Kyoto weather
    
    # Check tool results
    tool_messages = [msg for msg in converted_messages if msg["role"] == "tool"]
    assert len(tool_messages) == 3
    assert tool_messages[0]["tool_call_id"] == "weather_call_001"
    assert tool_messages[1]["tool_call_id"] == "news_call_002"
    assert tool_messages[2]["tool_call_id"] == "kyoto_weather_003"
    
    printer.print("Integration scenario test passed!")
    printer.print(f"Converted {len(converted_messages)} messages in weather assistant scenario")
    
    # Print scenario summary
    for i, (original, converted) in enumerate(zip(scenario_messages, converted_messages)):
        printer.print(f"Step {i+1}: {original.role.value} -> {converted['role']}")
        if original.role == Role.AI and any(isinstance(block, ToolCallBlock) for block in original.blocks):
            printer.print(f"  - Tool calls: {len(converted.get('tool_calls', []))}")
        elif original.role == Role.TOOL:
            printer.print(f"  - Tool call ID: {converted.get('tool_call_id', 'None')}")
            printer.print(f"  - Content preview: {converted.get('content', '')[:50]}...")


@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_server_enhanced_from_tool_call_method(llm: OpenAILlm):
    """
    Test the enhanced from_tool_call method that supports multiple tool calls with optional text.
    """
    printer.print("Testing enhanced from_tool_call method...")
    
    # Test 1: Single tool call with text
    printer.print("\n=== Test 1: Single tool call with text ===")
    message1 = Message.from_tool_call(
        tool_calls={
            "id": "call_123",
            "name": "get_weather",
            "arguments": {"city": "Tokyo", "unit": "celsius"}
        },
        text="I will check the weather for you.",
    )
    
    # Verify message structure
    assert message1.role == Role.AI
    assert len(message1.blocks) == 2
    assert isinstance(message1.blocks[0], TextBlock)
    assert isinstance(message1.blocks[1], ToolCallBlock)
    assert message1.blocks[0].text == "I will check the weather for you."
    assert message1.blocks[1].id == "call_123"
    assert message1.blocks[1].name == "get_weather"
    
    # Convert to OpenAI format
    converted1 = llm._convert_chat_completions_message(message1)
    assert converted1["role"] == "assistant"
    assert converted1["content"] == "I will check the weather for you."
    assert len(converted1["tool_calls"]) == 1
    assert converted1["tool_calls"][0]["id"] == "call_123"
    assert converted1["tool_calls"][0]["function"]["name"] == "get_weather"
    
    printer.print("✓ Single tool call with text test passed")
    
    # Test 2: Multiple tool calls with text
    printer.print("\n=== Test 2: Multiple tool calls with text ===")
    message2 = Message.from_tool_call(
        tool_calls=[
            {
                "id": "call_124",
                "name": "get_weather",
                "arguments": {"city": "Tokyo", "unit": "celsius"}
            },
            {
                "id": "call_125",
                "name": "get_news",
                "arguments": {"topic": "weather", "city": "Tokyo"}
            },
            {
                "id": "call_126",
                "name": "get_attractions",
                "arguments": {"city": "Tokyo", "limit": 5}
            }
        ],
        text="I will get weather, news, and attractions for you.",
    )
    
    # Verify message structure
    assert message2.role == Role.AI
    assert len(message2.blocks) == 4  # 1 TextBlock + 3 ToolCallBlocks
    assert isinstance(message2.blocks[0], TextBlock)
    assert message2.blocks[0].text == "I will get weather, news, and attractions for you."
    
    # Verify all tool calls
    tool_call_blocks = [block for block in message2.blocks if isinstance(block, ToolCallBlock)]
    assert len(tool_call_blocks) == 3
    assert tool_call_blocks[0].id == "call_124"
    assert tool_call_blocks[0].name == "get_weather"
    assert tool_call_blocks[1].id == "call_125"
    assert tool_call_blocks[1].name == "get_news"
    assert tool_call_blocks[2].id == "call_126"
    assert tool_call_blocks[2].name == "get_attractions"
    
    # Convert to OpenAI format
    converted2 = llm._convert_chat_completions_message(message2)
    assert converted2["role"] == "assistant"
    assert converted2["content"] == "I will get weather, news, and attractions for you."
    assert len(converted2["tool_calls"]) == 3
    
    # Verify tool call IDs and names
    tool_call_ids = [tc["id"] for tc in converted2["tool_calls"]]
    tool_call_names = [tc["function"]["name"] for tc in converted2["tool_calls"]]
    assert "call_124" in tool_call_ids
    assert "call_125" in tool_call_ids
    assert "call_126" in tool_call_ids
    assert "get_weather" in tool_call_names
    assert "get_news" in tool_call_names
    assert "get_attractions" in tool_call_names
    
    printer.print("✓ Multiple tool calls with text test passed")
    
    # Test 3: Multiple tool calls without text
    printer.print("\n=== Test 3: Multiple tool calls without text ===")
    message3 = Message.from_tool_call(
        tool_calls=[
            {
                "id": "call_127",
                "name": "get_time",
                "arguments": {"timezone": "UTC"}
            },
            {
                "id": "call_128",
                "name": "get_date",
                "arguments": {"format": "iso"}
            }
        ],
    )
    
    # Verify message structure
    assert message3.role == Role.AI
    assert len(message3.blocks) == 2  # Only ToolCallBlocks, no TextBlock
    assert all(isinstance(block, ToolCallBlock) for block in message3.blocks)
    
    # Verify tool calls
    assert message3.blocks[0].id == "call_127"
    assert message3.blocks[0].name == "get_time"
    assert message3.blocks[1].id == "call_128"
    assert message3.blocks[1].name == "get_date"
    
    # Convert to OpenAI format
    converted3 = llm._convert_chat_completions_message(message3)
    assert converted3["role"] == "assistant"
    assert converted3["content"] == ""  # Empty content when no text
    assert len(converted3["tool_calls"]) == 2
    
    printer.print("✓ Multiple tool calls without text test passed")
    
    # Test 4: Edge case - empty tool calls list
    printer.print("\n=== Test 4: Edge case - empty tool calls list ===")
    try:
        message4 = Message.from_tool_call(
            tool_calls=[],
            text="This should not work.",
        )
        # This should not raise an error, but should create a message with only text
        assert len(message4.blocks) == 1
        assert isinstance(message4.blocks[0], TextBlock)
        assert message4.blocks[0].text == "This should not work."
        
        converted4 = llm._convert_chat_completions_message(message4)
        assert converted4["role"] == "assistant"
        assert converted4["content"] == "This should not work."
        assert "tool_calls" not in converted4
        
        printer.print("✓ Empty tool calls list handled correctly")
    except Exception as e:
        printer.print(f"❌ Empty tool calls list failed: {e}")
        raise
    
    # Test 5: Complex arguments
    printer.print("\n=== Test 5: Complex arguments ===")
    message5 = Message.from_tool_call(
        tool_calls=[
            {
                "id": "call_129",
                "name": "search_hotels",
                "arguments": {
                    "city": "Tokyo",
                    "check_in": "2024-03-15",
                    "check_out": "2024-03-18",
                    "guests": 2,
                    "preferences": {
                        "amenities": ["wifi", "pool", "gym"],
                        "price_range": {"min": 100, "max": 300},
                        "rating": 4.0
                    }
                }
            }
        ],
        text="I will search for hotels in Tokyo with your preferences.",
    )
    
    # Verify complex arguments are preserved
    tool_call_block = message5.blocks[1]  # Second block should be the tool call
    assert tool_call_block.arguments["city"] == "Tokyo"
    assert tool_call_block.arguments["guests"] == 2
    assert tool_call_block.arguments["preferences"]["amenities"] == ["wifi", "pool", "gym"]
    assert tool_call_block.arguments["preferences"]["price_range"]["min"] == 100
    
    # Convert to OpenAI format
    converted5 = llm._convert_chat_completions_message(message5)
    assert converted5["role"] == "assistant"
    assert converted5["content"] == "I will search for hotels in Tokyo with your preferences."
    assert len(converted5["tool_calls"]) == 1
    
    # Verify arguments are properly serialized
    import json
    serialized_args = json.loads(converted5["tool_calls"][0]["function"]["arguments"])
    assert serialized_args["city"] == "Tokyo"
    assert serialized_args["guests"] == 2
    assert serialized_args["preferences"]["amenities"] == ["wifi", "pool", "gym"]
    
    printer.print("✓ Complex arguments test passed")
    
    printer.print("\n=== Enhanced from_tool_call method test summary ===")
    printer.print("✓ Single tool call with text")
    printer.print("✓ Multiple tool calls with text")
    printer.print("✓ Multiple tool calls without text")
    printer.print("✓ Empty tool calls list handling")
    printer.print("✓ Complex arguments support")
    printer.print("✓ OpenAI format conversion")
    
    printer.print("\nEnhanced from_tool_call method test completed successfully!")


@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_server_flexible_from_tool_call_parameters(llm: OpenAILlm):
    """
    Test the flexible from_tool_call method that supports various parameter types.
    """
    printer.print("Testing flexible from_tool_call method with various parameter types...")
    
    # Test 1: Single ToolCallBlock
    printer.print("\n=== Test 1: Single ToolCallBlock ===")
    tool_call_block = ToolCallBlock(
        id="call_133",
        name="get_weather",
        arguments={"city": "Tokyo", "unit": "celsius"}
    )
    
    message1 = Message.from_tool_call(
        tool_calls=tool_call_block,
        text="I will check the weather for you.",
    )
    
    # Verify message structure
    assert message1.role == Role.AI
    assert len(message1.blocks) == 2
    assert isinstance(message1.blocks[0], TextBlock)
    assert isinstance(message1.blocks[1], ToolCallBlock)
    assert message1.blocks[1].id == "call_133"
    assert message1.blocks[1].name == "get_weather"
    
    # Convert to OpenAI format
    converted1 = llm._convert_chat_completions_message(message1)
    assert converted1["role"] == "assistant"
    assert converted1["content"] == "I will check the weather for you."
    assert len(converted1["tool_calls"]) == 1
    assert converted1["tool_calls"][0]["id"] == "call_133"
    
    printer.print("✓ Single ToolCallBlock test passed")
    
    # Test 2: List of ToolCallBlocks
    printer.print("\n=== Test 2: List of ToolCallBlocks ===")
    tool_call_blocks = [
        ToolCallBlock(
            id="call_134",
            name="get_weather",
            arguments={"city": "Tokyo", "unit": "celsius"}
        ),
        ToolCallBlock(
            id="call_135",
            name="get_news",
            arguments={"topic": "weather", "city": "Tokyo"}
        ),
        ToolCallBlock(
            id="call_136",
            name="get_attractions",
            arguments={"city": "Tokyo", "limit": 5}
        )
    ]
    
    message2 = Message.from_tool_call(
        tool_calls=tool_call_blocks,
        text="I will get weather, news, and attractions for you.",
    )
    
    # Verify message structure
    assert message2.role == Role.AI
    assert len(message2.blocks) == 4  # 1 TextBlock + 3 ToolCallBlocks
    assert isinstance(message2.blocks[0], TextBlock)
    
    # Verify all tool calls
    tool_call_blocks_result = [block for block in message2.blocks if isinstance(block, ToolCallBlock)]
    assert len(tool_call_blocks_result) == 3
    assert tool_call_blocks_result[0].id == "call_134"
    assert tool_call_blocks_result[0].name == "get_weather"
    assert tool_call_blocks_result[1].id == "call_135"
    assert tool_call_blocks_result[1].name == "get_news"
    assert tool_call_blocks_result[2].id == "call_136"
    assert tool_call_blocks_result[2].name == "get_attractions"
    
    # Convert to OpenAI format
    converted2 = llm._convert_chat_completions_message(message2)
    assert converted2["role"] == "assistant"
    assert converted2["content"] == "I will get weather, news, and attractions for you."
    assert len(converted2["tool_calls"]) == 3
    
    printer.print("✓ List of ToolCallBlocks test passed")
    
    # Test 3: Mixed list (dicts and ToolCallBlocks)
    printer.print("\n=== Test 3: Mixed list (dicts and ToolCallBlocks) ===")
    mixed_tool_calls = [
        {
            "id": "call_137",
            "name": "get_weather",
            "arguments": {"city": "Tokyo", "unit": "celsius"}
        },
        ToolCallBlock(
            id="call_138",
            name="get_news",
            arguments={"topic": "weather", "city": "Tokyo"}
        ),
        {
            "id": "call_139",
            "name": "get_attractions",
            "arguments": {"city": "Tokyo", "limit": 5}
        }
    ]
    
    message3 = Message.from_tool_call(
        tool_calls=mixed_tool_calls,
        text="I will get weather, news, and attractions for you.",
    )
    
    # Verify message structure
    assert message3.role == Role.AI
    assert len(message3.blocks) == 4  # 1 TextBlock + 3 ToolCallBlocks
    assert isinstance(message3.blocks[0], TextBlock)
    
    # Verify all tool calls
    tool_call_blocks_result = [block for block in message3.blocks if isinstance(block, ToolCallBlock)]
    assert len(tool_call_blocks_result) == 3
    assert tool_call_blocks_result[0].id == "call_137"
    assert tool_call_blocks_result[0].name == "get_weather"
    assert tool_call_blocks_result[1].id == "call_138"
    assert tool_call_blocks_result[1].name == "get_news"
    assert tool_call_blocks_result[2].id == "call_139"
    assert tool_call_blocks_result[2].name == "get_attractions"
    
    # Convert to OpenAI format
    converted3 = llm._convert_chat_completions_message(message3)
    assert converted3["role"] == "assistant"
    assert converted3["content"] == "I will get weather, news, and attractions for you."
    assert len(converted3["tool_calls"]) == 3
    
    # Verify tool call IDs and names
    tool_call_ids = [tc["id"] for tc in converted3["tool_calls"]]
    tool_call_names = [tc["function"]["name"] for tc in converted3["tool_calls"]]
    assert "call_137" in tool_call_ids
    assert "call_138" in tool_call_ids
    assert "call_139" in tool_call_ids
    assert "get_weather" in tool_call_names
    assert "get_news" in tool_call_names
    assert "get_attractions" in tool_call_names
    
    printer.print("✓ Mixed list test passed")
    
    # Test 4: Error handling - invalid format
    printer.print("\n=== Test 4: Error handling - invalid format ===")
    try:
        # This should raise a ValueError
        Message.from_tool_call(
            tool_calls="invalid_format",
            text="This should fail.",
        )
        printer.print("❌ Error handling test failed - should have raised ValueError")
        assert False, "Should have raised ValueError for invalid format"
    except ValueError as e:
        printer.print(f"✓ Error handling test passed - caught ValueError: {e}")
    except Exception as e:
        printer.print(f"❌ Unexpected error: {e}")
        raise
    
    # Test 5: Complex ToolCallBlock with nested arguments
    printer.print("\n=== Test 5: Complex ToolCallBlock with nested arguments ===")
    complex_tool_call = ToolCallBlock(
        id="call_140",
        name="search_hotels",
        arguments={
            "city": "Tokyo",
            "check_in": "2024-03-15",
            "check_out": "2024-03-18",
            "guests": 2,
            "preferences": {
                "amenities": ["wifi", "pool", "gym"],
                "price_range": {"min": 100, "max": 300},
                "rating": 4.0
            }
        }
    )
    
    message5 = Message.from_tool_call(
        tool_calls=complex_tool_call,
        text="I will search for hotels in Tokyo with your preferences.",
    )
    
    # Verify complex arguments are preserved
    tool_call_block = message5.blocks[1]  # Second block should be the tool call
    assert tool_call_block.arguments["city"] == "Tokyo"
    assert tool_call_block.arguments["guests"] == 2
    assert tool_call_block.arguments["preferences"]["amenities"] == ["wifi", "pool", "gym"]
    assert tool_call_block.arguments["preferences"]["price_range"]["min"] == 100
    
    # Convert to OpenAI format
    converted5 = llm._convert_chat_completions_message(message5)
    assert converted5["role"] == "assistant"
    assert converted5["content"] == "I will search for hotels in Tokyo with your preferences."
    assert len(converted5["tool_calls"]) == 1
    
    # Verify arguments are properly serialized
    import json
    serialized_args = json.loads(converted5["tool_calls"][0]["function"]["arguments"])
    assert serialized_args["city"] == "Tokyo"
    assert serialized_args["guests"] == 2
    assert serialized_args["preferences"]["amenities"] == ["wifi", "pool", "gym"]

# Tests for parameter validation functionality

@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_parameter_validation_with_configuration(llm: OpenAILlm):
    """
    Test parameter validation when using configuration with default model.
    """
    
    # Create LLM with configuration that has default model
    config = OpenAIConfiguration(model=_model_name, temperature=0.7)
    configured_llm = get_configuration_llm(config)
    
    # Test that chat works without specifying model (uses config default)
    response = configured_llm.chat(
        messages=[Message.from_text(text="Hello, how are you?", role=Role.USER)]
    )
    assert response.message.role == Role.AI
    assert response.message.content is not None
    
    # Test that stream works without specifying model
    stream_response = configured_llm.stream(
        messages=[Message.from_text(text="Hello, how are you?", role=Role.USER)]
    )
    result = ""
    for chunk in stream_response:
        result += chunk.delta
    assert len(result) > 0
    

def test_openai_parameter_validation_missing_model_error():
    """
    Test that missing model parameter raises appropriate error.
    """
    from bridgic.llms.openai.openai_llm import OpenAIConfiguration
    
    # Create LLM without default model in configuration
    config = OpenAIConfiguration(temperature=0.7)  # No model specified
    configured_llm = OpenAILlm(api_key="mock-api-key", configuration=config)
    
    # Test that chat fails without model
    with pytest.raises(ValueError, match="Missing required parameters: model"):
        configured_llm.chat(
            messages=[Message.from_text(text="Hello, how are you?", role=Role.USER)]
        )
    
    # Test that stream fails without model (when iterating)
    with pytest.raises(ValueError, match="Missing required parameters: model"):
        stream_result = configured_llm.stream(
            messages=[Message.from_text(text="Hello, how are you?", role=Role.USER)]
        )
        # Stream is a generator, need to iterate to trigger validation
        next(stream_result)
    
    # Test that achat fails without model
    with pytest.raises(ValueError, match="Missing required parameters: model"):
        import asyncio
        asyncio.run(configured_llm.achat(
            messages=[Message.from_text(text="Hello, how are you?", role=Role.USER)]
        ))
    
    # Test that astream fails without model (when iterating)
    with pytest.raises(ValueError, match="Missing required parameters: model"):
        import asyncio
        async def test_astream():
            stream_result = configured_llm.astream(
                messages=[Message.from_text(text="Hello, how are you?", role=Role.USER)]
            )
            # Astream is an async generator, need to iterate to trigger validation
            await stream_result.__anext__()
        asyncio.run(test_astream())
    
    printer.print("✓ Missing model parameter error handling test passed")


@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_parameter_validation_method_override_config(llm: OpenAILlm):
    """
    Test that method parameters override configuration parameters.
    """
    
    # Create LLM with configuration that has default model
    config = OpenAIConfiguration(model="gpt-3.5-turbo", temperature=0.5)
    configured_llm = get_configuration_llm(config)
    
    # Test that method parameter overrides config model
    response = configured_llm.chat(
        model=_model_name,  # Override config model
        temperature=0.8,   # Override config temperature
        messages=[Message.from_text(text="Hello, how are you?", role=Role.USER)]
    )
    assert response.message.role == Role.AI
    assert response.message.content is not None
    
    # Test that None model parameter still uses config default
    response2 = configured_llm.chat(
        model=None,  # Should use config default
        messages=[Message.from_text(text="Hello, how are you?", role=Role.USER)]
    )
    assert response2.message.role == Role.AI
    assert response2.message.content is not None
    
    printer.print("✓ Method parameter override test passed")


@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_parameter_validation_structured_output(llm: OpenAILlm):
    """
    Test parameter validation for structured output methods.
    """
    
    class SimpleResponse(BaseModel):
        answer: str = Field(description="The answer to the question")
    
    # Create LLM with configuration
    config = OpenAIConfiguration(model=_model_name, temperature=0.7)
    configured_llm = get_configuration_llm(config)
    
    # Test structured_output with config model
    response = configured_llm.structured_output(
        constraint=PydanticModel(model=SimpleResponse),
        messages=[Message.from_text(text="What is 2+2?", role=Role.USER)]
    )
    assert isinstance(response, SimpleResponse)
    assert response.answer is not None
    
    # Test structured_output with method override
    response2 = configured_llm.structured_output(
        model=_model_name,  # Explicit model
        constraint=PydanticModel(model=SimpleResponse),
        messages=[Message.from_text(text="What is 3+3?", role=Role.USER)]
    )
    assert isinstance(response2, SimpleResponse)
    assert response2.answer is not None
    
    # Test that missing model raises error
    config_no_model = OpenAIConfiguration(temperature=0.7)  # No model
    llm_no_model = get_configuration_llm(config_no_model)
    
    with pytest.raises(ValueError, match="Missing required parameters: model"):
        llm_no_model.structured_output(
            constraint=PydanticModel(model=SimpleResponse),
            messages=[Message.from_text(text="What is 4+4?", role=Role.USER)]
        )


@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_parameter_validation_tool_selection(llm: OpenAILlm, tools):
    """
    Test parameter validation for tool selection methods.
    """
    
    # Create LLM with configuration
    config = OpenAIConfiguration(model=_model_name, temperature=0.7)
    configured_llm = get_configuration_llm(config)
    
    # Test select_tool with config model
    tool_calls, content = configured_llm.select_tool(
        tools=tools,
        messages=[Message.from_text(text="Get weather for Tokyo", role=Role.USER)]
    )
    assert isinstance(tool_calls, list)
    assert len(tool_calls)
    
    # Test select_tool with method override
    tool_calls2, content2 = configured_llm.select_tool(
        model=_model_name,  # Explicit model
        tools=tools,
        messages=[Message.from_text(text="Get news about technology", role=Role.USER)]
    )
    assert isinstance(tool_calls2, list)
    assert len(tool_calls2) > 0
    
    # Test that missing model raises error
    config_no_model = OpenAIConfiguration(temperature=0.7)  # No model
    llm_no_model = get_configuration_llm(config_no_model)
    
    with pytest.raises(ValueError, match="Missing required parameters: model"):
        llm_no_model.select_tool(
            tools=tools,
            messages=[Message.from_text(text="Get weather for Tokyo", role=Role.USER)]
        )

@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_parameter_validation_edge_cases():
    """
    Test edge cases for parameter validation.
    """
    
    # Test with empty configuration
    empty_config = OpenAIConfiguration()
    empty_llm = get_configuration_llm(empty_config)
    
    # Should fail because no model is provided
    with pytest.raises(ValueError, match="Missing required parameters: model"):
        empty_llm.chat(
            messages=[Message.from_text(text="Hello", role=Role.USER)]
        )
    
    # Test with model=None explicitly
    with pytest.raises(ValueError, match="Missing required parameters: model"):
        empty_llm.chat(
            model=None,
            messages=[Message.from_text(text="Hello", role=Role.USER)]
        )
    
    # Test with empty messages (should still validate model)
    with pytest.raises(ValueError, match="Missing required parameters: model"):
        empty_llm.chat(
            messages=[]
        )
@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
@pytest.mark.asyncio
async def test_openai_parameter_validation_edge_cases_async():
    """
    Test edge cases for parameter validation.
    """
    
    # Test with empty configuration
    empty_config = OpenAIConfiguration()
    empty_llm = get_configuration_llm(empty_config)
    
    # Should fail because no model is provided
    with pytest.raises(ValueError, match="Missing required parameters: model"):
        result = await empty_llm.achat(
            messages=[Message.from_text(text="Hello", role=Role.USER)]
        )
    
    # Test with model=None explicitly
    with pytest.raises(ValueError, match="Missing required parameters: model"):
        result = await empty_llm.achat(
            model=None,
            messages=[Message.from_text(text="Hello", role=Role.USER)]
        )
    
    # Test with empty messages (should still validate model)
    with pytest.raises(ValueError, match="Missing required parameters: model"):
        result = await empty_llm.achat(
            messages=[]
        )

@pytest.mark.skipif(
    (_api_key is None) or (_model_name is None),
    reason="OPENAI_API_KEY or OPENAI_MODEL_NAME is not set",
)
def test_openai_parameter_validation_error_messages():
    """
    Test that error messages are clear and informative.
    """
    
    # Test single missing parameter
    config = OpenAIConfiguration(temperature=0.7)
    configured_llm = get_configuration_llm(config)
    
    try:
        configured_llm.chat(
            messages=[Message.from_text(text="Hello", role=Role.USER)]
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        error_msg = str(e)
        assert "Missing required parameters:" in error_msg
        assert "model" in error_msg
        assert "," not in error_msg  # Single parameter, no comma