"""Tests for the LLM TabularAgent module."""

from unittest.mock import MagicMock, patch

from iter8ml.services.llm import LLMAgentConfig, TabularAgent


def _mock_litellm_response(content: str = "Test LLM response") -> MagicMock:
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = content
    return mock


def _setup_litellm_mock(return_value=None):
    mock_mod = MagicMock()
    mock_mod.completion.return_value = return_value or _mock_litellm_response()
    return mock_mod


class TestTabularAgent:
    def test_disabled_agent_returns_empty_commentary(self):
        agent = TabularAgent(LLMAgentConfig(enabled=False))
        result = agent.explain_shap([], "test_model", "classification")
        assert result.content == ""

    def test_disabled_performance_returns_empty(self):
        agent = TabularAgent(LLMAgentConfig(enabled=False))
        result = agent.explain_performance({"roc_auc": 0.9}, "test_model", "classification")
        assert result.content == ""

    def test_disabled_feature_summary_returns_empty(self):
        agent = TabularAgent(LLMAgentConfig(enabled=False))
        result = agent.summarize_features([{"feature_name": "a", "importance": 0.5}])
        assert result.content == ""

    def test_enabled_without_litellm_returns_placeholder(self):
        agent = TabularAgent(LLMAgentConfig(enabled=True))
        with patch.dict("sys.modules", {"litellm": None}):
            result = agent.explain_performance({"roc_auc": 0.9}, "test_model", "classification")
        assert "litellm" in result.content

    def test_enabled_calls_litellm_completion(self):
        mock_mod = _setup_litellm_mock()
        agent = TabularAgent(LLMAgentConfig(enabled=True, model="gpt-4o"))
        with patch.dict("sys.modules", {"litellm": mock_mod}):
            result = agent.explain_performance({"roc_auc": 0.9}, "test_model", "classification")

        assert result.content == "Test LLM response"
        mock_mod.completion.assert_called_once()
        call_kwargs = mock_mod.completion.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"

    def test_enabled_with_api_key_env(self):
        mock_mod = _setup_litellm_mock(_mock_litellm_response("Response"))
        agent = TabularAgent(LLMAgentConfig(enabled=True, api_key_env="TEST_API_KEY"))
        with (
            patch.dict("sys.modules", {"litellm": mock_mod}),
            patch.dict("os.environ", {"TEST_API_KEY": "test-key-123"}),
        ):
            result = agent.explain_shap([], "model", "classification")

        assert result.content == "Response"
        call_kwargs = mock_mod.completion.call_args[1]
        assert call_kwargs["api_key"] == "test-key-123"

    def test_enabled_with_missing_api_key_returns_placeholder(self):
        mock_mod = _setup_litellm_mock()
        agent = TabularAgent(LLMAgentConfig(enabled=True, api_key_env="NONEXISTENT_KEY_12345"))
        with patch.dict("sys.modules", {"litellm": mock_mod}):
            result = agent.explain_performance({"roc_auc": 0.9}, "test_model", "classification")
        assert "API key" in result.content

    def test_enabled_with_api_base(self):
        mock_mod = _setup_litellm_mock()
        agent = TabularAgent(LLMAgentConfig(enabled=True, api_base="http://localhost:11434"))
        with patch.dict("sys.modules", {"litellm": mock_mod}):
            agent.explain_shap([], "model", "classification")

        call_kwargs = mock_mod.completion.call_args[1]
        assert call_kwargs["api_base"] == "http://localhost:11434"

    def test_config_defaults(self):
        config = LLMAgentConfig()
        assert config.enabled is False
        assert config.model == "claude-sonnet-4-20250514"
        assert config.api_key_env == ""
        assert config.api_base is None
