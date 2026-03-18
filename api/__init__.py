"""
api/__init__.py
===============
Package initialiser for the FastAPI application layer.

Exposes the FastAPI  app  instance so it can be imported and served with:

    uvicorn api:app --reload

or referenced in tests:

    from api import app
"""

# from api.main import app  # uncomment once main.py is implemented

__all__: list[str] = [
    # "app",
]
