# Bridgic: The Next Generation of Agent Development

[![PyPI](https://img.shields.io/pypi/v/bridgic.svg)](https://pypi.org/search/?q=bridgic)
[![License](https://img.shields.io/pypi/l/bridgic.svg)](https://pypi.org/project/bridgic/)
[![GitHub](https://img.shields.io/github/stars/bitsky-tech/bridgic?style=social)](https://github.com/bitsky-tech/bridgic)

The name **Bridgic** embodies our core philosophy **"Bridging Logic and Magic"** where:

- *Logic* represents structured and predictable execution flows, forming the foundation of reliable systems.
- *Magic* refers to the autonomous parts that can make dynamic decisions and solve problems creatively.


## Installation

Bridgic requires Python 3.9 or newer. Make sure your environment meets this requirement before installing.

=== "pip"

    ```bash
    pip install bridgic
    ```

=== "uv"

    ```bash
    uv add bridgic
    ```

After installation, verify that Bridgic is installed correctly:

=== "pip"

    ```bash
    python -c "from bridgic.core import __version__; print(f'Bridgic version: {__version__}')"
    ```

=== "uv"

    ```bash
    uv run python -c "from bridgic.core import __version__; print(f'Bridgic version: {__version__}')"
    ```


## What is the Vision

Our vision is to make intelligent systems easy to build for everyone.

Bridgic isn't just a framework—it's defining how the next generation of intelligent systems will be built.

By fundamentally rethinking the relationship between deterministic workflows and autonomous agents and establishing a unified runtime foundation, we’re setting a new standard for the development of agentic systems.

This is not merely a design choice—it is a **fundamental redefinition** of what is possible in agentic system development. We are creating a world where developers can seamlessly transition from **deterministic workflows** to **autonomous agents** within a single, unified framework.


## How We Achieve It

To realize our vision of making intelligent systems easy to build for everyone, Bridgic is built on four fundamental technological innovations that collectively redefine what is possible in agentic system development:

**🎯 Innovative Dynamic Directed Graph Engine**: At the core lies **Dynamic Directed Graph (DDG)**—an innovative runtime model that fundamentally breaks down the artificial barriers between traditional static graph systems and autonomous agent construction. Unlike static graph systems, DDG operates as a dynamic runtime model that enables intelligent systems to adapt their execution structure in real-time, providing developers with a **unified development experience** that seamlessly spans the entire spectrum from structured workflows to fully autonomous systems.

**🤝 Powerful Human-in-the-Loop APIs**: Bridgic provides powerful human-in-the-loop APIs that enable sophisticated interaction patterns between intelligent systems and human operators. These APIs support interrupting execution units at any point in the workflow, allowing systems to pause, request external feedback, and seamlessly resume execution with the acquired feedback, which supports an agentic system to leverage human's input when needed.

**🔌 Seamless Third-Party Integration**: Bridgic establishes a comprehensive integration architecture through well-designed internal abstractions, including:

- **technology-neutral model integration** that enables seamless integration with any LLM provider
- **systematic MCP integration** that transforms tool integration into a composition opportunity
- **seamless enterprise observability integration** that ensures agentic systems are transparent and optimizable.

**🎨 High-Information-Density Agent Structure Language**: Bridgic introduces **Agent Structure Language (ASL)**—a Python-native DSL that enables developers to express sophisticated agentic structures within **limited code footprint**. This code organization paradigm is **optimized for the future of AI-assisted development**, making it particularly well-suited for AI code generation.


Ready to bridge logic and magic? Explore the documentation to continue your journey:

* **[Tutorials](tutorials/index.md)** — Learn through hands-on examples
* **[Understanding](understanding/introduction.md)** — Deep dive into core concepts
* **[API Reference](reference/bridgic-core/bridgic/core/agentic/index.md)** — Complete API documentation