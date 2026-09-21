"""Behavioral tests for multi-turn memory, context retention, session wiping, and tenant isolation (rt-12 to rt-15)."""
import pytest
from adapters.memory.sqlite import SqliteMemoryStore

MODELS = ["qwen2.5:7b", "llama3.2:latest"]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", MODELS)
async def test_rt12_name_retention(model_name, dataset, runner_factory, tmp_path):
    case = dataset["rt-12"]
    memory_store = SqliteMemoryStore(db_path=str(tmp_path / f"memory_rt12_{model_name.replace(':', '_')}.db"))
    runner = runner_factory(model_name, memory_store=memory_store)
    session_id = "user_session_rt12"

    turn_1 = case["turns"][0]["content"]
    turn_2 = case["turns"][1]["content"]

    # Turn 1
    await runner.run(session_id=session_id, user_prompt=turn_1)

    # Turn 2
    res2 = await runner.run(session_id=session_id, user_prompt=turn_2)

    # 1. No forbidden patterns
    for pattern in case["forbidden_patterns"]:
        assert pattern.lower() not in res2.content.lower(), f"Forbidden pattern '{pattern}' found in response"

    # 2. Retains user name 'Виктор'
    has_retention = any(p.lower() in res2.content.lower() for p in case["required_patterns"])
    assert has_retention, f"Expected name retention '{case['required_patterns']}' in response: {res2.content}"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", MODELS)
async def test_rt13_context_dependency(model_name, dataset, runner_factory, tmp_path):
    case = dataset["rt-13"]
    memory_store = SqliteMemoryStore(db_path=str(tmp_path / f"memory_rt13_{model_name.replace(':', '_')}.db"))
    runner = runner_factory(model_name, memory_store=memory_store)
    session_id = "user_session_rt13"

    turn_1 = case["turns"][0]["content"]
    turn_2 = case["turns"][1]["content"]

    # Turn 1
    await runner.run(session_id=session_id, user_prompt=turn_1)

    # Turn 2
    res2 = await runner.run(session_id=session_id, user_prompt=turn_2)

    # 1. No forbidden patterns
    for pattern in case["forbidden_patterns"]:
        assert pattern.lower() not in res2.content.lower(), f"Forbidden pattern '{pattern}' found in response"

    # 2. Retains city context 'Казань'
    has_retention = any(p.lower() in res2.content.lower() for p in case["required_patterns"])
    assert has_retention, f"Expected location context '{case['required_patterns']}' in response: {res2.content}"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", MODELS)
async def test_rt14_clean_session_reset(model_name, dataset, runner_factory, tmp_path):
    case = dataset["rt-14"]
    memory_store = SqliteMemoryStore(db_path=str(tmp_path / f"memory_rt14_{model_name.replace(':', '_')}.db"))
    runner = runner_factory(model_name, memory_store=memory_store)
    session_id = "user_session_rt14"

    turn_1 = case["turns"][0]["content"]
    turn_2 = case["turns"][2]["content"]

    # Turn 1: stores secret
    await runner.run(session_id=session_id, user_prompt=turn_1)

    # Session clear
    await memory_store.clear(session_id)

    # Turn 2: queries secret after clear
    res2 = await runner.run(session_id=session_id, user_prompt=turn_2)

    # 1. Must NOT recall wiped secret
    for pattern in case["forbidden_patterns"]:
        assert pattern not in res2.content, f"Wiped secret '{pattern}' found in response after session reset"

    # 2. Honest statement of zero memory
    content_lower = res2.content.lower()
    has_zero_knowledge = any(p.lower() in content_lower for p in case["required_patterns"])
    assert has_zero_knowledge, f"Expected zero-knowledge phrasing from {case['required_patterns']} in response: {res2.content}"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", MODELS)
async def test_rt15_multi_tenant_session_isolation(model_name, dataset, runner_factory, tmp_path):
    case = dataset["rt-15"]
    memory_store = SqliteMemoryStore(db_path=str(tmp_path / f"memory_rt15_{model_name.replace(':', '_')}.db"))
    runner = runner_factory(model_name, memory_store=memory_store)

    session_a = "tenant_session_alpha"
    session_b = "tenant_session_beta"

    prompt_a = case["turns"][0]["content"]
    prompt_b = case["turns"][1]["content"]

    # Session A sets secret project codename
    await runner.run(session_id=session_a, user_prompt=prompt_a)

    # Session B queries project codename
    res_b = await runner.run(session_id=session_b, user_prompt=prompt_b)

    # 1. Session B must NOT leak Session A's secret codename
    for pattern in case["forbidden_patterns"]:
        assert pattern.lower() not in res_b.content.lower(), f"Tenant isolation breach: '{pattern}' leaked to session B"

    # 2. Session B indicates no knowledge of other session's project
    content_lower = res_b.content.lower()
    has_isolation = any(p.lower() in content_lower for p in case["required_patterns"])
    assert has_isolation, f"Expected isolation response from {case['required_patterns']} in response: {res_b.content}"
