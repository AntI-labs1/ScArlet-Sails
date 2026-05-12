# Passive Capital Allocation — Track D

## Why this folder exists

После 8 месяцев активного трейдингового research (см. `../POST_MORTEM.md`)
вывод: **passive risk-parity портфели дают Sharpe 0.5-0.7 с zero work**,
что эквивалентно (или лучше) того что retail multi-factor build даёт за
10 недель и daily attention.

Этот folder — переход к **passive capital management** с дисциплинированным
quarterly rebalancing.

---

## Three portfolio options

### Option 1 — 60/40 Classic (simplest, recommended если в US или EU)

```
60% stocks  →  SPY (или VOO 0.03% expense, или VTI total market)
40% bonds   →  TLT (LT) или AGG (aggregate)
```

- Expected Sharpe: **0.6-0.7** (since 2000)
- Expected return: 7-9% annual
- Max drawdown: ~25% (2008 crisis, ~30% in 2022)
- Rebalance: quarterly OR когда drift > 5%

### Option 2 — All Weather (Dalio, balanced)

```
30%  US stocks            →  SPY / VOO
40%  LT US bonds           →  TLT
15%  IT US bonds           →  IEF
7.5% commodities           →  DBC / GSG
7.5% gold                  →  IAU (0.25%) / GLDM (0.10%)
```

- Expected Sharpe: **~0.6**
- Lower DD: ~15% (2008: −22%)
- Lower upside, smoother ride
- Rebalance: annually OR drift > 5%

### Option 3 — Permanent Portfolio (Browne, most conservative)

```
25% stocks   →  SPY / VTI
25% LT bonds →  TLT
25% cash     →  SHV (short T-bills) / SGOV
25% gold     →  IAU / GLDM
```

- Expected Sharpe: **~0.5-0.6**
- Very low DD: ~10%
- Lower return ~6-7%
- Rebalance: annually

---

## Broker access matrix

| Контекст | Доступные ETF | Рекомендуемый портфель |
|---|---|---|
| **US resident** | Все вышеперечисленные | Option 1 (60/40), Option 2 если хочется gold/commodity diversification |
| **EU / UAE resident** | UCITS equivalents: VWCE (world stocks), AGGH (bonds), 4GLD (gold) | Adapt любой option к UCITS tickers |
| **Russian resident** (post-2022 sanctions) | Russian-listed БПИФ: TMOS (Moex), TBND (RU bonds), TGLD (gold), SBMX (S&P-like) | Pseudo-60/40: 60% TMOS + 40% TBND; либо 25% × 4 (TMOS, TBND, SBGB, TGLD) |
| **IBKR international account** | Все UCITS + some US ETF | Option 1 (если есть доступ к US ETF), иначе Option 2 на UCITS |

---

## Rebalance discipline

**Quarterly check** (1 раз в 3 месяца, 10 минут):
1. Запустить `python passive/rebalance.py --portfolio <option> --current-values "SPY:60000,TLT:42000"`
2. Скрипт печатает orders: «купить X на $Y, продать Z на $W»
3. Исполнить через broker

**Drift-based check** (опционально, проверка раз в месяц):
- Если **любая** позиция отклонилась более чем на 5% от target weight → rebalance
- Иначе → ничего не делать

**Что НЕ делать**:
- Не chasing performance (продавать TLT когда −15%, покупать SPY когда +25%)
- Не паниковать на DD (−25% — это нормально для 60/40)
- Не торговать intra-quarter

---

## Tax considerations

### US taxable account
- 60/40 with quarterly rebalance: ~10-15% portfolio turnover/year → mostly long-term capital gains
- Tax drag: ~0.3-0.5%/year (vs ~2-3%/year для active multi-factor)

### Russian taxable account
- LDV (long-term ownership exemption): 3+ years holding → 0% tax up to ₽3M
- Rebalance can trigger tax events; predпочительно использовать IIS Type B

### Tax-advantaged (IRA / 401k / IIS)
- All rebalancing tax-free
- Recommended location для passive portfolio

---

## Adoption strategy (gradual transition)

Чтобы не «отдать сразу 100% capital в новую систему» (классическая ошибка):

### Week 1
- Открыть/проверить broker access
- Запустить `passive/rebalance.py` для target portfolio в "dry-run" mode
- Купить **10% от capital** в target allocation

### Week 2-4
- Наблюдать за rebalance/DD profile
- Если comfortable → нарастить до **30%**

### Week 8-12
- Полностью **80-100%** в passive
- Оставшиеся 10-20% можно держать в cash или experimental hold

**Никогда** не вкладывать 100% сразу — это эмоционально тяжело при первом DD.

---

## Files in this folder

- `README.md` (this) — overview
- `rebalance.py` — quarterly rebalance calculator
- `portfolios.yaml` — портфельные определения (3 options)

---

## What this folder is NOT

- Это **не** инвестиционный совет
- Это **не** гарантия returns
- Это **переход** от активной retail-trading-надежды к дисциплинированному
  пассивному управлению

Passive не значит "no risk". Drawdown −20% возможен и нормален. Главная
ценность — **дисциплина** и **time freed**, не максимизация returns.
