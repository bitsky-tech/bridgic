import pytest

from bridgic.core.utils.console import printer

@pytest.mark.asyncio
async def test_printer():
    printer.print([])
    printer.print("Hello Bridgic!", color="red")
    printer.print("Hello Bridgic!", color="green")
    printer.print("Hello Bridgic!", color="yellow")
    printer.print("Hello Bridgic!", color="blue")
    printer.print("Hello Bridgic!", color="purple")
    printer.print("Hello Bridgic!", color="cyan")
