"""The Check door (HTTP API). Run with: uvicorn moderation.api:create_app --factory"""

from moderation.api.app import create_app

__all__ = ["create_app"]
