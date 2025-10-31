# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-10-31

### Added
- Initial release of agent-tracer
- Framework-agnostic tracing API
- Support for LangSmith tracer
- Support for LangFuse tracer  
- Support for LangWatch tracer
- Support for Arize Phoenix tracer
- Support for Opik tracer
- Support for Traceloop tracer
- Async/await support for modern Python applications
- Automatic sensitive data masking (API keys, passwords, etc.)
- Hierarchical tracing (workflows → agents → tools)
- LangChain callback integration
- Context manager API for easy step tracing
- Comprehensive examples for different use cases
- Full type hints and mypy support

### Features
- `TracingService` - Main API for tracing
- `BaseTracer` - Abstract base class for tracer implementations
- `Log` schema for structured logging
- `TracingConfig` for service configuration
- Automatic tracer initialization based on environment variables
- Background worker for async trace processing
- Error handling and recovery

### Documentation
- Comprehensive README with quick start guide
- API reference documentation
- Integration examples for LangChain
- Custom framework integration examples
- Multi-agent system examples
- Contributing guidelines

## [Unreleased]

### Planned
- Support for more tracing backends
- Performance optimizations
- Enhanced error reporting
- CLI tools for configuration
- Web UI for trace visualization
- Integration guides for popular frameworks

