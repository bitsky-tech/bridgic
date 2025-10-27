import pytest
import os
from typing import Dict, List, Any

from bridgic.core.automa import GraphAutoma, worker, Snapshot
from bridgic.core.automa.interaction import Event, InteractionFeedback, InteractionException
from bridgic.core.model.protocols import ToolSelection
from bridgic.core.agentic.tool_specs import as_tool
from bridgic.core.automa.args import System
from bridgic.core.agentic import ReActAutoma
from bridgic.llms.openai import OpenAILlm, OpenAIConfiguration
from bridgic.llms.vllm import VllmServerLlm, VllmServerConfiguration
from tests.core.model.mock_llm import MockLlm

_openai_api_key = os.environ.get("OPENAI_API_KEY")
_openai_model_name = os.environ.get("OPENAI_MODEL_NAME", default="gpt-5-mini")
_openai_api_base = os.environ.get("OPENAI_BASE_URL", default=None)

_vllm_api_base = os.environ.get("VLLM_SERVER_API_BASE")
_vllm_api_key = os.environ.get("VLLM_SERVER_API_KEY", default="EMPTY")
_vllm_model_name = os.environ.get("VLLM_SERVER_MODEL_NAME", default="Qwen/Qwen3-4B-Instruct-2507")

@pytest.fixture
def llm() -> ToolSelection:
    # Use OpenAI LLM by setting environment variables:
    # export OPENAI_API_KEY="xxx"
    # export OPENAI_MODEL_NAME="xxx"
    if _openai_api_key:
        print(f"\nUsing `OpenAILlm` ({_openai_model_name}) to test ReactAutoma...")
        return OpenAILlm(
            api_key=_openai_api_key,
            configuration=OpenAIConfiguration(model=_openai_model_name),
            timeout=10,
            api_base=_openai_api_base,
        )
    # Use VLLM Server LLM by setting environment variables:
    # export VLLM_SERVER_API_KEY="xxx"
    # export VLLM_SERVER_API_BASE="xxx"
    # export VLLM_SERVER_MODEL_NAME="xxx"
    if _vllm_api_base:
        print(f"\nUsing `VllmServerLlm` ({_vllm_model_name}) to test ReactAutoma...")
        return VllmServerLlm(
            api_key=_vllm_api_key,
            api_base=_vllm_api_base,
            configuration=VllmServerConfiguration(model=_vllm_model_name),
            timeout=10,
        )

    print(f"\nUsing `MockLlm` to test ReactAutoma...")
    return MockLlm()


################################################################################
# Test Case 1.
# Configurations:
# - single tool `get_weather`.
# - tools are provided at initialization.
# - inputs are whole `messages` list.
# - functional tool `get_weather`, async def.
################################################################################

async def get_weather(
    city: str,
) -> str:
    """
    Retrieves current weather for the given city.

    Parameters
    ----------
    city : str
        The city to get the weather of, e.g. New York.
    
    Returns
    -------
    str
        The weather for the given city.
    """
    # Mock the weather API call.
    return f"The weather in {city} is sunny today and the temperature is 20 degrees Celsius."

@pytest.fixture
def react_automa_1(llm: ToolSelection) -> ReActAutoma:
    return ReActAutoma(
        llm=llm,
        tools=[get_weather],
    )

@pytest.mark.asyncio
async def test_react_automa_case_1(react_automa_1: ReActAutoma):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "What is the weather in Tokyo?",
        }
    ]
    result = await react_automa_1.arun(messages=messages)
    assert bool(result)
    assert "sunny" in result
    assert "20" in result
    assert "not able to select tools" not in result


################################################################################
# Test Case 2.
# Configurations:
# - single tool `multiply`.
# - tools are provided at runtime.
# - inputs use `user_msg` and `chat_history`.
# - Automa tool, non-async method `multiply()`.
################################################################################

def multiply(x: int, y: int) -> int:
    """
    This function is used to multiply two numbers.

    Parameters
    ----------
    x : int
        The first number to multiply
    y : int
        The second number to multiply

    Returns
    -------
    int
        The product of the two numbers
    """
    # Note: this function need not to be implemented.
    ...

@as_tool(multiply)
class MultiplyAutoma(GraphAutoma):
    @worker(is_start=True, is_output=True)
    def multiply(self, x: int, y: int):
        return x * y

@pytest.fixture
def react_automa_2(llm: ToolSelection) -> ReActAutoma:
    return ReActAutoma(
        llm=llm,
        system_prompt="You are a helpful assistant that are good at calculating by using tools.",
    )

@pytest.mark.asyncio
async def test_react_automa_case_2(react_automa_2: ReActAutoma):
    result = await react_automa_2.arun(
        user_msg="What is 235 * 4689?",
        chat_history=[
            {
                "role": "user",
                "content": "Could you help me to do some calculations?",
            },
            {
                "role": "assistant",
                "content": "Of course, I can help you with that.",
            }
        ],
        # tools are provided at runtime here.
        tools=[MultiplyAutoma],
    )
    assert bool(result)
    assert "1101915" in result or "1,101,915" in result
    assert "not able to select tools" not in result


################################################################################
# Test Case 3: Customer Refund Application Processing with Human-in-the-Loop using ReAct
# Scenario: AI Agent processes customer refund applications, requiring human confirmation at the final step
# Using ReAct mode, extracting workers as tools, obtaining automa instance through System("automa") parameter injection
################################################################################


# Tool 1: Read refund emails
async def read_refund_emails() -> List[Dict[str, Any]]:
    """
    Read all emails with "refund" in the subject from the mailbox
    
    Returns
    -------
    List[Dict[str, Any]]
        List of refund emails, containing subject, content, order_id, customer_email and other information
    """
    # Simulate reading refund emails from mailbox
    emails = [
        {
            "subject": "Refund Application - Order #12345",
            "content": "Hello, I would like to apply for a refund, order number is 12345, the reason is product quality issues.",
            "order_id": "12345",
            "customer_email": "customer1@example.com"
        },
        {
            "subject": "Refund Application - Order #67890", 
            "content": "Hello, I would like to apply for a refund, order number is 67890, the reason is size not suitable.",
            "order_id": "67890",
            "customer_email": "customer2@example.com"
        }
    ]
    print(f"📧 Read {len(emails)} refund emails")
    return emails

# Tool 2: Query order information
async def query_order_info(order_id: str) -> Dict[str, Any]:
    """
    Query refund amount and payment method from backend system based on order number
    
    Parameters
    ----------
    order_id : str
        Order number
        
    Returns
    -------
    Dict[str, Any]
        Order information, containing order_id, amount, payment_method, customer_name and other information
    """
    # Simulate querying order information from backend system
    order_info = {
        "order_id": order_id,
        "amount": 5000.0 if order_id == "12345" else 2500.0,  # Simulate different amounts
        "payment_method": "Alipay" if order_id == "12345" else "WeChat Pay",
        "customer_name": f"Customer{order_id}"
    }
    print(f"🔍 Queried order {order_id}: Amount {order_info['amount']} yuan, Payment method {order_info['payment_method']}")
    
    return order_info

# Tool 3: Generate reply and application for a single order
async def generate_reply_and_application(order_id: str, amount: float, payment_method: str, customer_name: str) -> str:
    """
    Generate a polite reply and fill out the refund application form
    
    Parameters
    ----------
    order_id : str
        Order number
    amount : float
        Refund amount
    payment_method : str
        Payment method
    customer_name : str
        Customer name
        
    Returns
    -------
    str
        Generated reply email content
    """
    # Generate polite reply
    reply = f"""
Dear {customer_name},

Thank you for contacting us. We have received your refund application (Order Number: {order_id}).

Refund Amount: {amount} yuan
Payment Method: {payment_method}

We will process your refund application within 3-5 business days, and the refund will be returned to your {payment_method} account through the original payment method.

If you have any questions, please feel free to contact us.

Best regards
Customer Service Team
    """.strip()
    
    print(f"📝 Generated reply and application form for order {order_id}")
    
    return reply

# Tool 4: Save to draft folder
async def save_to_draft_folder(reply: str, order_id: str, amount: float, customer_email: str) -> str:
    """
    Save email draft and refund application form to the "Drafts" folder of customer service mailbox
    
    Parameters
    ----------
    reply : str
        Reply email content
    order_id : str
        Order number
    amount : float
        Refund amount
    customer_email : str
        Customer email
        
    Returns
    -------
    str
        Save confirmation message
    """
    print("📁 Saved email draft and refund application form to customer service mailbox 'Drafts' folder")
    
    return f"Saved refund email draft for order {order_id}, amount {amount} yuan, recipient {customer_email}"

# Tool 5: Human approval (with Human-in-the-Loop)
async def human_approval(order_id: str, amount: float, customer_email: str, automa = System("automa")) -> str:
    """
    Human approval for sending refund email - Human-in-the-Loop intervention point
    
    Parameters
    ----------
    order_id : str
        Order number
    amount : float
        Refund amount
    customer_email : str
        Customer email
    automa : System("automa")
        Automa instance obtained through parameter injection
        
    Returns
    -------
    str
        Processing result
    """
    print(f"🔄 Human approval required for order {order_id}, amount {amount} yuan, recipient {customer_email}")
    # Create human approval event
    event = Event(
        event_type="refund_approval",
        data={
            "prompt_to_user": f"""
🔄 Refund application processing completed, human approval required

📊 Processing Summary:
- Order Number: {order_id}
- Refund Amount: {amount} yuan
- Recipient: {customer_email}

⚠️  Please carefully verify the amount and recipient information, then select an action after confirmation:

🟢 Confirm Send - Send refund email
🔴 Return for Modification - Let AI reprocess
            """.strip(),
            "approval_info": {
                "order_id": order_id,
                "amount": amount,
                "customer_email": customer_email
            }
        }
    )
    
    # Trigger human interaction
    feedback: InteractionFeedback = automa.interact_with_human(event)
    
    if feedback.data == "confirm":
        print("✅ Human approval: Send refund email")
        return "Refund email sent, all applications processed"
    elif feedback.data == "reject":
        print("❌ Human rejection: Need reprocessing")
        return "Refund application returned, needs reprocessing"
    else:
        print("⚠️ Unknown feedback, default to return")
        return "Refund application returned, needs reprocessing"


# Create ReAct test case
@pytest.fixture
def refund_react_automa(llm: ToolSelection) -> ReActAutoma:
    """Refund processing ReAct Automa instance"""
    return ReActAutoma(
        llm=llm,
        system_prompt="""You are a professional customer service refund processing assistant. You need to process customer refund applications according to the following steps:

1. First call read_refund_emails() to read refund emails from the mailbox
2. For each email, call query_order_info(order_id) to query order information
3. Call generate_reply_and_application(order_info) to generate reply and application form
4. Call save_to_draft_folder(reply, application) to save to draft folder
5. Finally call human_approval(draft_data) for human confirmation

Please follow this process step by step, ensuring each step is accurate. Note: Each tool needs to be called separately, cannot be processed in batch.

Once all the orders have been processed, the model needs to summarize the results, in the following format (see the following example) :

- Order number: xxx, approved: Yes, refund: xxx yuan
- Order number: yyy, approved: No, refund: 0 yuan
""",
        tools=[
            read_refund_emails,
            query_order_info, 
            generate_reply_and_application,
            save_to_draft_folder,
            human_approval
        ]
    )


# Shared fixtures for all test cases.
@pytest.fixture(scope="session")
def tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("tmp_path")

@pytest.mark.skipif(
    not ((_openai_api_key is not None and _openai_model_name is not None) or (_vllm_api_base is not None and _vllm_api_key is not None and _vllm_model_name is not None)),
    reason="Either OpenAI (OPENAI_API_KEY and OPENAI_MODEL_NAME) or VLLM (VLLM_SERVER_API_BASE, VLLM_SERVER_API_KEY, and VLLM_SERVER_MODEL_NAME) must be fully configured.",
)
@pytest.mark.asyncio
async def test_refund_react_processing_with_human_approval(refund_react_automa: ReActAutoma, request, tmp_path):
    """Test refund processing workflow in ReAct mode, triggering human approval"""
    try:
        result = await refund_react_automa.arun(
            user_msg="Please help me process customer refund applications, need to read emails, query orders, generate replies, and finally require human confirmation."
        )
    except InteractionException as e:
        # Verify interaction exception - should have two interactions (processed two orders)
        assert len(e.interactions) == 2
        for interaction in e.interactions:
            assert interaction.event.event_type == "refund_approval"
            assert "Refund application processing completed" in interaction.event.data["prompt_to_user"]
            assert "Confirm Send" in interaction.event.data["prompt_to_user"]
            assert "Return for Modification" in interaction.event.data["prompt_to_user"]
        
        # Verify snapshot
        assert type(e.snapshot.serialized_bytes) is bytes
        
        # Save interaction IDs and snapshot (simulate persistence)
        interaction_ids = [interaction.interaction_id for interaction in e.interactions]
        request.config.cache.set("interaction_ids", interaction_ids)
        # Save snapshot to temporary file (simulate persistence)
        snapshot_file = tmp_path / "refund_snapshot.bytes"
        version_file = tmp_path / "refund_snapshot.version"
        snapshot_file.write_bytes(e.snapshot.serialized_bytes)
        version_file.write_text(e.snapshot.serialization_version)
        
        print(f"🔄 ReAct refund processing paused, waiting for human approval (Interaction ID: {interaction.interaction_id})")

@pytest.fixture
def refund_react_automa_deserialized(tmp_path):
    """Restore refund processing ReAct Automa from snapshot"""
    snapshot_file = tmp_path / "refund_snapshot.bytes"
    version_file = tmp_path / "refund_snapshot.version"
    
    serialized_bytes = snapshot_file.read_bytes()
    serialization_version = version_file.read_text()
    
    snapshot = Snapshot(
        serialized_bytes=serialized_bytes,
        serialization_version=serialization_version
    )
    
    # Restore ReAct Automa from snapshot
    deserialized_automa = ReActAutoma.load_from_snapshot(snapshot)
    assert type(deserialized_automa) is ReActAutoma
    return deserialized_automa

@pytest.fixture
def refund_approval_feedback(request):
    """Human approval feedback - confirm send"""
    interaction_ids = request.config.cache.get("interaction_ids", None)
    confirm_feedback = InteractionFeedback(
        interaction_id=interaction_ids[0],
        data="confirm"
    )
    reject_feedback = InteractionFeedback(
        interaction_id=interaction_ids[1],
        data="reject"
    )
    return [confirm_feedback, reject_feedback]

@pytest.mark.skipif(
    not ((_openai_api_key is not None and _openai_model_name is not None) or (_vllm_api_base is not None and _vllm_api_key is not None and _vllm_model_name is not None)),
    reason="Either OpenAI (OPENAI_API_KEY and OPENAI_MODEL_NAME) or VLLM (VLLM_SERVER_API_BASE, VLLM_SERVER_API_KEY, and VLLM_SERVER_MODEL_NAME) must be fully configured.",
)
@pytest.mark.asyncio
async def test_refund_approval_feedback_results(refund_approval_feedback, refund_react_automa_deserialized):
    """Test scenario of human approval and sending in ReAct mode"""
    result = await refund_react_automa_deserialized.arun(
        interaction_feedbacks=refund_approval_feedback
    )
    
    assert "12345, approved: Yes" in result
    assert "67890, approved: No" in result
    print("✅ Test passed: Both the approval and denial of human intervention take effect")