from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import tmcra_v4_slow_graph as slow_graph
from tmcra_service.planner_provider import recall_planner_route
from tmcra_service.qwen36_planner_adapter import LocalQwenRecallRolePlanner
from tmcra_service.api_models import ProjectionBuildProgressResponse
from tmcra_service.session_graph import LocalSessionGraphAgent
from tmcra_service.writer import LeasedDeepSeekClient
from tmcra_service.writer_provider import (
    OPENAI_COMPATIBLE_PROVIDER,
    primary_writer_route,
    reviewer_writer_route,
)


CUSTOM_MODEL = "operator-model-v1"
CUSTOM_BASE_URL = "http://127.0.0.1:22435/v1"


class ConfigurableLocalModelTests(unittest.TestCase):
    def test_projection_progress_accepts_dedicated_local_slot(self) -> None:
        progress = ProjectionBuildProgressResponse(
            schema_version="tmcra.projection-build-progress.1",
            scope_name="test-scope",
            status="running",
            stage="session_maps",
            progress_percent=10,
            completed_units=1,
            total_units=10,
            session_maps={},
            session_atlas={},
            visual_atlas={},
            knowledge_base={},
            detail="building",
            updated_at=1.0,
            agent_enabled=True,
            resource_isolation="dedicated-local-slot",
        )

        self.assertEqual(progress.resource_isolation, "dedicated-local-slot")

    def test_provider_routes_accept_operator_selected_model_identities(self) -> None:
        environment = {
            "TMCRA_WRITER_PROVIDER": "deepseek",
            "TMCRA_WRITER_BASE_URL": "https://models.example.invalid/v1",
            "TMCRA_WRITER_MODEL": "operator-writer-v1",
            "TMCRA_WRITER_API_KEY_POOL": "operator-test-key",
            "TMCRA_WRITER_PROMPT_ADAPTER": "none",
            "TMCRA_WRITER_REVIEWER_PROVIDER": "deepseek",
            "TMCRA_WRITER_REVIEWER_MODEL": "operator-reviewer-v2",
            "TMCRA_RECALL_PLANNER_PROVIDER": "deepseek",
            "TMCRA_RECALL_PLANNER_MODEL": "operator-planner-v3",
            "TMCRA_RECALL_PLANNER_PROMPT_ADAPTER": "none",
        }
        self.assertEqual(primary_writer_route(environment).model, "operator-writer-v1")
        self.assertEqual(
            reviewer_writer_route(environment).model, "operator-reviewer-v2"
        )
        self.assertEqual(recall_planner_route(environment).model, "operator-planner-v3")

    def test_writer_and_reviewer_accept_a_custom_local_model_alias(self) -> None:
        environment = {
            "TMCRA_WRITER_PROVIDER": "local-qwen",
            "TMCRA_WRITER_BASE_URL": CUSTOM_BASE_URL,
            "TMCRA_WRITER_MODEL": CUSTOM_MODEL,
            "TMCRA_WRITER_API_KEY_POOL": "local-test-key",
            "TMCRA_WRITER_PROMPT_ADAPTER": "qwen36-v5",
            "TMCRA_WRITER_REVIEWER_PROVIDER": "local-qwen",
            "TMCRA_WRITER_REVIEWER_PROMPT_ADAPTER": "qwen36-reconciliation-v1",
        }
        primary = primary_writer_route(environment)
        reviewer = reviewer_writer_route(environment)
        self.assertEqual(primary.model, CUSTOM_MODEL)
        self.assertEqual(primary.base_url, CUSTOM_BASE_URL)
        self.assertEqual(reviewer.model, CUSTOM_MODEL)

    def test_planner_accepts_a_custom_local_model_alias(self) -> None:
        environment = {
            "TMCRA_WRITER_BASE_URL": CUSTOM_BASE_URL,
            "TMCRA_WRITER_MODEL": CUSTOM_MODEL,
            "TMCRA_WRITER_API_KEY_POOL": "local-test-key",
            "TMCRA_RECALL_PLANNER_PROVIDER": "local-qwen",
            "TMCRA_RECALL_PLANNER_PROMPT_ADAPTER": "qwen36-planner-v1",
        }
        route = recall_planner_route(environment)
        planner = LocalQwenRecallRolePlanner(
            base_url=route.base_url,
            model=route.model,
            api_keys=route.api_keys,
        )
        self.assertEqual(planner.model, CUSTOM_MODEL)
        self.assertEqual(planner.base_url, CUSTOM_BASE_URL)

    def test_openai_compatible_writer_accepts_an_arbitrary_model_identity(self) -> None:
        route = primary_writer_route(
            {
                "TMCRA_WRITER_PROVIDER": OPENAI_COMPATIBLE_PROVIDER,
                "TMCRA_WRITER_BASE_URL": "https://models.example.invalid/v1",
                "TMCRA_WRITER_MODEL": CUSTOM_MODEL,
                "TMCRA_WRITER_API_KEY_POOL": "operator-test-key",
                "TMCRA_WRITER_PROMPT_ADAPTER": "openai-memory-v1",
            }
        )
        self.assertEqual(route.model, CUSTOM_MODEL)
        client = LeasedDeepSeekClient(
            v4=object(),
            pool=SimpleNamespace(lease_seconds=300),
            operation_id="test-operation",
            base_url=route.base_url,
            model=route.model,
            timeout=1,
            max_tokens=256,
            provider=route.provider,
            prompt_adapter=route.prompt_adapter,
        )
        self.assertEqual(client.model, CUSTOM_MODEL)

    def test_slow_graph_accepts_a_custom_local_model_alias(self) -> None:
        environment = {
            "TMCRA_SLOW_GRAPH_BASE_URL": CUSTOM_BASE_URL,
            "TMCRA_SLOW_GRAPH_MODEL": CUSTOM_MODEL,
            "TMCRA_SLOW_GRAPH_API_KEY_POOL": "local-test-key",
            "TMCRA_SLOW_GRAPH_MAX_TOKENS": "4096",
            "TMCRA_SLOW_GRAPH_PROMPT_ADAPTER": "qwen36-slow-graph-v1",
        }
        with patch.dict(os.environ, environment, clear=False):
            config = slow_graph._local_qwen_config()
        self.assertEqual(config.model, CUSTOM_MODEL)
        self.assertEqual(config.base_url, CUSTOM_BASE_URL)

    def test_session_graph_inherits_the_configured_local_model(self) -> None:
        agent = LocalSessionGraphAgent.from_env(
            {
                "TMCRA_SESSION_GRAPH_PROVIDER": "local-qwen",
                "TMCRA_SESSION_GRAPH_API_KEY": "local-test-key",
                "TMCRA_LOCAL_WRITER_BASE_URL": CUSTOM_BASE_URL,
                "TMCRA_LOCAL_WRITER_MODEL": CUSTOM_MODEL,
            }
        )
        self.assertIsNotNone(agent)
        assert agent is not None
        self.assertEqual(agent.model, CUSTOM_MODEL)
        self.assertEqual(agent.base_url, CUSTOM_BASE_URL)


if __name__ == "__main__":
    unittest.main()
