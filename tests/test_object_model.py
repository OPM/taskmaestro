"""Tests for ObjectModel generic base."""

from __future__ import annotations

from pydantic import BaseModel

from taskekrabbe import ObjectModel


class FakeClient:
    """A non-Pydantic arbitrary object for testing."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port


class TestObjectModelBasic:
    """Basic construction and field access."""

    def test_wrap_arbitrary_object(self) -> None:
        client = FakeClient("localhost", 8080)
        model = ObjectModel(value=client)
        assert model.value is client
        assert model.value.host == "localhost"
        assert model.value.port == 8080

    def test_wrap_primitive(self) -> None:
        model = ObjectModel(value=42)
        assert model.value == 42


class TestObjectModelTypeAlias:
    """Type alias usage (simple wrappers)."""

    def test_type_alias(self) -> None:
        WrappedClient = ObjectModel[FakeClient]
        client = FakeClient("10.0.0.1", 443)
        model = WrappedClient(value=client)
        assert model.value is client
        assert model.value.host == "10.0.0.1"


class TestObjectModelSubclass:
    """Subclass with extra fields."""

    def test_subclass_with_extra_fields(self) -> None:
        class EnrichedClient(ObjectModel[FakeClient]):
            timeout: float
            retries: int

        client = FakeClient("db.local", 5432)
        model = EnrichedClient(value=client, timeout=30.0, retries=3)
        assert model.value is client
        assert model.timeout == 30.0
        assert model.retries == 3

    def test_subclass_is_base_model(self) -> None:
        class Wrapped(ObjectModel[FakeClient]):
            label: str

        client = FakeClient("h", 1)
        model = Wrapped(value=client, label="test")
        assert isinstance(model, BaseModel)
        assert isinstance(model, ObjectModel)
