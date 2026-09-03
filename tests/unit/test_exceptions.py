import pytest

from core.exceptions import (
    DomainError,
    LLMPluginError,
    LLMConnectionError,
    LLMTimeoutError,
    LLMResponseError,
)


class TestDomainExceptions:
    def test_hierarchy(self):
        assert issubclass(DomainError, Exception)
        assert issubclass(LLMPluginError, DomainError)
        assert issubclass(LLMConnectionError, LLMPluginError)
        assert issubclass(LLMTimeoutError, LLMPluginError)
        assert issubclass(LLMResponseError, LLMPluginError)

    def test_domain_error_catching(self):
        with pytest.raises(DomainError):
            raise LLMConnectionError("Cannot connect")

        with pytest.raises(LLMPluginError):
            raise LLMTimeoutError("Request timed out")

    def test_llm_response_error_attributes(self):
        err_with_status = LLMResponseError("Internal Server Error", status_code=500)
        assert str(err_with_status) == "Internal Server Error"
        assert err_with_status.status_code == 500

        err_without_status = LLMResponseError("Malformed response")
        assert str(err_without_status) == "Malformed response"
        assert err_without_status.status_code is None
