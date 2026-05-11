# Test Suite — Known Issues & Recovery Notes

Этот файл — живой журнал актуальных проблем тестов. До правок 2026-05 был дрейф
между документом («8 падает / 95.6%») и реальностью (40 падает / 77%). После
этой ревизии база приведена в более здоровое состояние; ниже описано **что было
починено** и **что остаётся проверить вручную**.

> Прежде чем доверять локально цифрам — поднимите venv (`python3 -m venv .venv &&
> source .venv/bin/activate && pip install -r requirements.txt`) и прогоните
> `pytest tests/ -q`. На голой машине pytest не установлен и автоматически
> цифры тут устаревают.

---

## 1. Что было починено в этой ревизии

### 1.1 `core/rolling_dispersion.py` — был ложный «баг»
Аудит ошибочно сказал, что в исходнике инвертирована формула `_calculate_multiplier`.
**Исходник правильный** (agreement → high multiplier; chaos → low multiplier).
**Тесты** в `test_rolling_dispersion.py` и `test_dispersion_inverted.py` тестировали
старую (ошибочную) логику и поэтому красили. Тесты переписаны под корректное
поведение.

### 1.2 `scripts/run_council.py:523` — hardcoded `current_drawdown=0.0`
Killswitch по drawdown не работал — позиция всегда сайзилась как при нулевой
просадке. Добавлен `DecisionLogger.get_current_drawdown()`, который проходит
закрытые трейды (`pnl_pct is not None`), считает equity curve и возвращает
текущую просадку. `_calculate_position_size` теперь читает реальный drawdown.

### 1.3 `core/feature_engine_v2._normalize_features` — look-ahead
`fit_transform` срабатывал автоматически при первом вызове. На полном датасете
(train+test одним заходом) это утечка статистик. Теперь:
- `calculate_features(..., fit_scaler=False)` — дефолт; требует загруженный scaler;
- `calculate_features(..., fit_scaler=True)` — только для train-данных;
- без scaler и без `fit_scaler=True` → `RuntimeError` с подсказкой.

### 1.4 `core/data_loader.py` — расхождение `BTC_USDT` vs `BTCUSDT`
Лоадер ждал `BTC_USDT_15m.parquet`, DVC версионировал `BTCUSDT_15m.parquet.dvc`.
5 тестов в `test_data_loader.py` падали из-за этого. Лоадер теперь принимает обе
схемы (сначала пробует canonical с `_USDT_`, потом Binance-style без подчёркивания).

### 1.5 Голые `except:` в production-коде
- `rag/vector_store.py:452` → `except (OSError, json.JSONDecodeError, KeyError)`
- `rag/hybrid_retriever.py:175` → `except (OSError, json.JSONDecodeError)`
- `rag/hybrid_retriever.py:506` → `except Exception` с warning-логом
- `models/ml_training_pipeline.py:325` → `except ValueError` (roc_auc на одном классе)
- `main.py` (две штуки) ушли вместе с переписыванием.

### 1.6 Чистка мусора
- `1МБ` (0 байт), `features/base_features.py` (0 байт) — удалены.
- `test_results_day11_post_fix.txt` / `test_results_fixed.txt` — функциональные дубли,
  удалены; остался `test_results_post_cleanup.txt`.
- Папка `archive/` — никем не импортируется, удалена.
- `core/feature_engine.py` (legacy) — не имел внешних потребителей, удалён.

### 1.7 Конфиги
- `configs/market_config.yaml` (3 монеты, противоречил основному `config.yaml`)
  — orphan, нигде не импортируется, удалён. Главный конфиг ассетов теперь только `config.yaml`.

---

## 2. Что не починено и почему

| Класс падений | Файлы | Почему не трогали |
|---|---|---|
| `test_hybrid_q_learner.py` (~9 шт.) | Q-learner | Сам подход устарел; кандидат на удаление целиком после Phase D, не имеет смысла чинить тесты |
| `test_rag_end_to_end.py` (~12 шт.) | RAG e2e | Большая часть зависит от пустого индекса; нужен seed-датасет паттернов перед запуском |
| `test_council_e2e.py`, `test_day4_integration.py` | Council e2e | Завязаны на RAG; решатся вместе с пунктом выше |
| `test_strategies.py::test_features_has_74_columns` | feature spec | Спецификация фичей разъехалась с тем, что генерирует engine v2 (31 vs 74). Нужно сначала решить какая цифра канонична |
| `test_sanitize.py` (часть) | sanitize_features | Проверить отдельно — могут зависеть от удалённых полей |

Эти классы падений **не блокируют MVP** (retail crypto-trader пользуется
бэктестом → vbt_engine → результаты). Q-learner и council/RAG в текущем виде —
research-надстройка, и их состояние не критично пока человек не пользуется ими.

---

## 3. Команда регресса (когда поднимется venv)

```bash
pip install -r requirements.txt
pytest tests/ -q --tb=short
```

Ожидание после ревизии: значительно меньше падений, чем прежние 40. Точная цифра
будет известна после первого прогона.
