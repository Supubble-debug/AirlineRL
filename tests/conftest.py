"""Test-session settings.

Protocol §13.1: ITSM env worker children must start with PYTHONHASHSEED=0. pytest itself has
already started (its own hashing stays random); setting the variable here makes every child the
tests spawn start with it, which is exactly the launcher contract. Tests that need an unfixed
child override it explicitly (monkeypatch / subprocess env).
"""

import os

os.environ["PYTHONHASHSEED"] = "0"
