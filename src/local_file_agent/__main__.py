"""Allow ``python -m local_file_agent`` to invoke the CLI."""

from local_file_agent.cli import app

if __name__ == "__main__":
    app()
