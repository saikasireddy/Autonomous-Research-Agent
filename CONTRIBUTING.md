# Contributing to Autonomous Research Agent

Thank you for your interest in contributing to this project!

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Autonomous-Research-Agent.git
   cd Autonomous-Research-Agent
   ```

3. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Install Ollama and pull the model:
   ```bash
   ollama pull llama3.1
   ```

## Running Tests

```bash
pytest tests/
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings to all functions and classes
- Keep functions focused and single-purpose

## Pull Request Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit:
   ```bash
   git add .
   git commit -m "Add: description of your changes"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a Pull Request on GitHub

## Commit Message Convention

- `Add:` for new features
- `Fix:` for bug fixes
- `Update:` for improvements to existing features
- `Refactor:` for code restructuring
- `Docs:` for documentation changes
- `Test:` for test additions/modifications

## Areas for Contribution

- **Agent Improvements**: Enhance researcher, analyzer, comparator, or synthesizer agents
- **New Features**: Add support for more paper sources, export formats, etc.
- **Performance**: Optimize FAISS indexing, chunk retrieval, or LLM calls
- **Testing**: Add unit tests, integration tests
- **Documentation**: Improve README, add tutorials, create examples
- **Bug Fixes**: Address issues from the issue tracker

## Questions?

Open an issue on GitHub for any questions or discussions.
