# 📊 Token Cost & Context Optimization Benchmark Report (Phase 2)

**Date:** `2026-09-06 17:24:33 UTC`  
**Model:** `qwen2.5:7b`  
**Scope:** 20 Representative Multi-Turn Autonomous Agent Tasks  
**Optimization Mode:** Observation Compactor (250 chars) + Sliding Dialogue Window (10 msgs) + Minified Schemas  

---

## 🏆 Executive Summary & Acceptance Criteria Verification

| Acceptance Criterion | Target Metric | Baseline (Phase 1.5) | Phase 2 (Optimized) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Token Volume Reduction** | **>= 30.0%** (<= 66,770) | 95,386 tokens | **59,129 tokens** (**38.0% reduction**) | ✅ **PASSED** |
| **Task Success Rate** | **100% (20/20)** | 20/20 (100%) | **20/20 (100%)** | ✅ **PASSED** |
| **Cumulative Inference Cost** | **<= $0.02340 USD** | $0.03344 | **$0.02059** (-38.4%) | ✅ **PASSED** |
| **Repeated Context Ratio** | **< 23.7%** | 23.7% | **40.2%** | ❌ FAILED |

---

## 📈 Side-by-Side Comparative Metrics (Baseline vs. Phase 2)

| Metric | Baseline (Phase 1.5) | Phase 2 (Optimized) | Delta | Change (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Input Tokens** | 90,035 | **55,963** | -34,072 | -37.8% |
| **Output Tokens** | 5,351 | **3,166** | -2,185 | -40.8% |
| **Total Tokens** | 95,386 | **59,129** | **-36,257** | **-38.0%** |
| **Repeated Context Tokens** | 21,383 | **22,499** | +1,116 | 5.2% |
| **Repeated Context %** | 23.7% | **40.2%** | +16.5% | — |
| **Estimated Cost (USD)** | $0.03344 | **$0.02059** | $-0.01285 | -38.4% |
| **Avg Latency per Task** | 6.64s | **38.96s** | +32.32s | — |
| **Successful Tasks** | 20/20 | **20/20** | 0 | 100% |

---

## 📋 Detailed Per-Task Comparison Breakdown

| # | Category | Prompt | Baseline Tokens | Optimized Tokens | Savings (%) | Baseline Rep % | Optimized Rep % | Status |
| :- | :--- | :--- | -: | -: | -: | -: | -: | :-: |
| 1 | Skills / Weather | `Выполни утреннюю сводку` | 4,731 | **7,603** | **+60.7%** | 61.2% | 68.4% | ✅ |
| 2 | Skills / Weather | `Какая сейчас погода в Минске?` | 4,573 | **5,407** | **+18.2%** | 17.8% | 60.3% | ✅ |
| 3 | Skills / Weather | `Выполни утреннюю сводку для города London` | 1,490 | **9,327** | **+526.0%** | 0.0% | 64.2% | ✅ |
| 4 | Skills / Weather | `Какая погода в Париже?` | 1,719 | **851** | **-50.5%** | 0.0% | 0.0% | ✅ |
| 5 | Skills / Weather | `Какая температура в Токио?` | 1,863 | **932** | **-50.0%** | 0.0% | 0.0% | ✅ |
| 6 | Skills / DevOps | `Проверь состояние системы и сервера` | 20,217 | **12,999** | **-35.7%** | 57.2% | 55.7% | ✅ |
| 7 | Skills / DevOps | `Какая операционная система и версия Python установлены?` | 2,117 | **2,216** | **+4.7%** | 0.0% | 33.0% | ✅ |
| 8 | Skills / DevOps | `Сколько свободного места на диске?` | 2,194 | **1,023** | **-53.4%** | 0.0% | 0.0% | ✅ |
| 9 | Skills / DevOps | `Работает ли локальный Ollama API? Проверь версию.` | 2,316 | **2,101** | **-9.3%** | 0.0% | 34.5% | ✅ |
| 10 | Skills / DevOps | `Покажи информацию о платформе и ресурсах хоста` | 2,519 | **1,137** | **-54.9%** | 0.0% | 0.0% | ✅ |
| 11 | Code / Tools | `Найди в кодовой базе, где определяется инструмент exec` | 10,736 | **1,118** | **-89.6%** | 46.5% | 0.0% | ✅ |
| 12 | Code / Tools | `Какие инструменты зарегистрированы в registry.py?` | 5,342 | **1,059** | **-80.2%** | 30.2% | 0.0% | ✅ |
| 13 | Code / Tools | `Найди в проекте файлы, где используется sqlite3` | 2,735 | **1,100** | **-59.8%** | 0.0% | 0.0% | ✅ |
| 14 | Code / Tools | `Проверь незакоммиченные изменения в git репозитории через git diff` | 3,088 | **1,327** | **-57.0%** | 0.0% | 0.0% | ✅ |
| 15 | Code / Tools | `Найди файл конфигурации config.py и скажи, какие переменные окружения поддерживаются` | 3,435 | **1,383** | **-59.7%** | 0.0% | 0.0% | ✅ |
| 16 | Conversational | `Расскажи кратко, какие у тебя есть возможности, инструменты и сценарии?` | 4,215 | **1,537** | **-63.5%** | 0.0% | 0.0% | ✅ |
| 17 | Conversational | `Объясни простыми словами, что такое ReAct агент и чем он отличается от обычного LLM чат-бота?` | 4,907 | **1,785** | **-63.6%** | 0.0% | 0.0% | ✅ |
| 18 | Conversational | `Напиши краткий чек-лист из 5 правил хорошего кода на Python` | 5,206 | **1,886** | **-63.8%** | 0.0% | 0.0% | ✅ |
| 19 | Conversational / Multi-turn | `Как применить эти правила к асинхронному коду на asyncio?` | 5,917 | **2,241** | **-62.1%** | 0.0% | 0.0% | ✅ |
| 20 | Conversational / Multi-turn | `Сформулируй краткое резюме из 2 предложений по нашим рекомендациям` | 6,066 | **2,097** | **-65.4%** | 0.0% | 0.0% | ✅ |

---

## 🔬 Technical Optimizations Implemented

1. **Observation Compactor (`core/runner.py`):**
   - Prior turn observations (`turn < current_turn - 1`) exceeding 250 characters are automatically compacted to concise excerpts with trailing markers.
   - The immediately preceding observation is preserved in full, preserving reasoning accuracy.
2. **Dialogue History Sliding Window (`core/runner.py` & `adapters/memory/sqlite.py`):**
   - Memory retrieval is bound by `history_limit=10` (5 full conversational turns).
   - Reverse SQL limit subquery ensures the newest messages are retrieved chronologically.
3. **Tool Schema & System Prompt Minification:**
   - Tool parameter JSON schemas are minified using `separators=(',', ':')` without whitespace.
   - Dense directive rules reduce prompt instruction token footprint by > 50%.

---
*Generated automatically on 2026-09-06 17:24:33 UTC via `scripts/run_benchmarks.py --mode optimized`.*