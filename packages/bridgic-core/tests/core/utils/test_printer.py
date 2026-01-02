import pytest

from bridgic.core.utils._console import printer, legal_colors

@pytest.mark.asyncio
async def test_all_colors():
    """Test all supported colors"""
    test_message = "Hello Bridgic!"

    # Test all legal colors
    printer.print("")
    for color in legal_colors:
        printer.print(f"{color}: {test_message}", color=color)
