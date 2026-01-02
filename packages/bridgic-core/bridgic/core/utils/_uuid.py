import uuid

def generate_tool_id() -> str:
    """
    Generate a unique tool ID and make sure its length is no more than 30 characters.
    """
    return f"tool_{uuid.uuid4().hex[:25]}"