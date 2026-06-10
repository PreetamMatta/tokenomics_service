# tokenomics — Model Economics Microservice

Provider-agnostic model catalog (OpenRouter + local YAML), quality enrichment
(Artificial Analysis + curated overrides), and transparent, Range-aware
monthly cost estimation for LLM workloads. Information layer only: no routing
decisions, no billing reconciliation, no usage tracking (v2).

## Quickstart

```bash
uv sync                                            # install deps (Python 3.11+)
uv run uvicorn --factory tokenomics.app:create_app # serve on :8000
open http://localhost:8000/docs                    # interactive OpenAPI UI
```

Or with Docker:

```bash
docker build -t tokenomics .
docker run -p 8000:8000 -e AA_API_KEY=$AA_API_KEY tokenomics
```

Environment variables:

| Variable | Purpose |
| --- | --- |
| `AA_API_KEY` | Artificial Analysis key (optional — without it the service still serves the catalog and cost estimation; capability tiers then come only from `config/overrides.yaml`) |
| `OPENROUTER_TTL_SECONDS` | OpenRouter catalog cache TTL (default 86400) |
| `AA_TTL_SECONDS` | Artificial Analysis cache TTL (default 86400) |
| `CONFIG_DIR` | Config directory (default `./config`) |

Configuration lives in `config/`: `settings.yaml` (URLs/TTLs), `tiers.yaml`
(percentile→tier thresholds), `model_aliases.yaml` (AA slug → OpenRouter id),
`overrides.yaml` (curated tiers/tags — always win), and `local_models/*.yaml`
(full ModelCards for self-hosted models).

All provider data sits behind in-memory TTL caches — there is no database.
`POST /v1/models/refresh` forces a re-fetch.

## API at a glance

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | liveness + per-provider freshness |
| GET | `/v1/capabilities` | capability enum, tier semantics, derivation thresholds |
| GET | `/v1/schedules/presets` | `light_4x6`, `standard_7x5`, `heavy_15x7`, `always_on_24x7` |
| GET | `/v1/models` | filterable ModelCard list |
| GET | `/v1/models/{model_id}` | single ModelCard (ids contain `/`) |
| POST | `/v1/models/refresh` | force provider re-fetch |
| GET | `/v1/models/queries/cheapest` | cheapest model meeting a capability/tier bar |
| GET | `/v1/models/queries/best` | highest tier, ties by evidence then price |
| GET | `/v1/models/queries/best-value` | top-5 by tier per blended dollar |
| POST | `/v1/estimates/cost` | one workload → CostBreakdown |
| POST | `/v1/estimates/pipeline` | Pipeline → PipelineCostBreakdown |
| POST | `/v1/estimates/compare` | one workload across N models, ranked |

Every selection-helper response includes a `justification` block (tier, price,
benchmarks cited) so a consuming agent can make its own decision. Every cost
response lists `estimated_fields` — the prices that used fallback defaults
(e.g. the 10%-of-input cache-read last resort).

Uncertain quantities are `Range` objects (`{"min": ..., "typical": ...,
"max": ...}`); any Range input propagates through every computation to
Range outputs.

## Worked example 1 — find the cheapest competent coding model

```bash
curl 'localhost:8000/v1/models?capability=coding&min_tier=3&max_input_price=5'
curl 'localhost:8000/v1/models/queries/cheapest?capability=coding&min_tier=3'
```

The `cheapest` response names the model, the blended price that won, how many
candidates were considered, and the evidence behind its tier.

## Worked example 2 — estimate one workload (the spec's anchor case)

A continuous stream at 2M input + 20K output tokens/hour with a 0.85 cache
hit ratio, on `standard_7x5` (H = 152.075 h/month), against a $3/$15 model
with $0.30/M cache reads, costs **$260.05/month**:

```bash
curl -X POST localhost:8000/v1/estimates/cost -H 'content-type: application/json' -d '{
  "workload": {
    "type": "continuous_stream",
    "role": "listener",
    "model_id": "anthropic/claude-sonnet-4.6",
    "input_tokens_per_hour": 2000000,
    "output_tokens_per_hour": 20000,
    "cache_hit_ratio": 0.85
  },
  "schedule": "standard_7x5"
}'
```

The breakdown shows the cache split (45.6M fresh / 258.5M cached input
tokens) and per-dimension costs.

## Worked example 3 — compare models on the same workload

```bash
curl -X POST localhost:8000/v1/estimates/compare -H 'content-type: application/json' -d '{
  "model_ids": ["anthropic/claude-sonnet-4.6", "openai/gpt-5.2-mini", "deepseek/deepseek-r1"],
  "workload": {
    "type": "per_request",
    "role": "api",
    "requests_per_period": 5000,
    "period": "day",
    "input_tokens_per_request": 1500,
    "output_tokens_per_request": 400
  },
  "schedule": "always_on_24x7"
}'
```

Returns one `CostBreakdown` per model plus `ranked_by_total` (cheapest first;
Ranges rank by their `typical` value).

## Worked example 4 — the Scaffle ambient pipeline (spec §9)

Four components on `standard_7x5`: an ambient listener (ContinuousStream), an
action executor (TriggeredAction), a code builder (BuildAction with
`iterations_per_task` as a Range — the main variance source), and a daily
memory job (RecurringJob). Replace the model ids with your preferred
cheap/mid/frontier picks from `/v1/models`:

```bash
curl -X POST localhost:8000/v1/estimates/pipeline -H 'content-type: application/json' -d '{
  "pipeline": {
    "name": "scaffle_ambient",
    "components": [
      {
        "type": "continuous_stream", "role": "ambient_listener",
        "model_id": "openai/gpt-5.2-mini",
        "input_tokens_per_hour": 1500000, "output_tokens_per_hour": 5000,
        "cache_hit_ratio": 0.90
      },
      {
        "type": "triggered_action", "role": "action_executor",
        "model_id": "anthropic/claude-sonnet-4.6",
        "triggers_per_hour": 2, "context_tokens_per_trigger": 100000,
        "output_tokens_per_trigger": 2000, "cache_hit_ratio": 0.2
      },
      {
        "type": "build_action", "role": "code_builder",
        "model_id": "anthropic/claude-opus-4.8",
        "tasks_per_period": 3, "period": "week",
        "iterations_per_task": {"min": 4, "typical": 12, "max": 40},
        "input_tokens_per_iteration": 60000,
        "output_tokens_per_iteration": 4000,
        "reasoning_tokens_per_iteration": 2000,
        "subagent_spawn_probability": 0.3,
        "subagent_overhead_multiplier": 1.5,
        "cache_hit_ratio": 0.7
      },
      {
        "type": "recurring_job", "role": "daily_memory",
        "model_id": "openai/gpt-5.2-mini",
        "frequency": "daily",
        "input_tokens_per_invocation": 200000,
        "output_tokens_per_invocation": 5000,
        "cache_hit_ratio": 0.5
      }
    ]
  },
  "schedule": "standard_7x5"
}'
```

The response is a `PipelineCostBreakdown` with four `CostBreakdown`s; the
`code_builder` totals are Ranges (the iteration Range propagates), so the
pipeline `total_cost` is a Range too.

The same call from Python, via the typed client:

```python
from tokenomics.client import TokenomicsClient

with TokenomicsClient("http://localhost:8000") as client:
    pipeline = {...}  # the JSON body above, under "pipeline"
    result = client.estimate_pipeline(pipeline, "standard_7x5")
    print(result.total_cost)            # Range(min=..., typical=..., max=...)
    for component in result.per_component:
        print(component.workload_role, component.total_cost, component.cost_per_action)
```

For library use without HTTP, `tokenomics.service.ModelEconomicsService`
exposes the same operations in-process (`build_default_service()` wires the
default provider stack).

## Development

```bash
uv run pytest        # 137 tests: engine math vs hand-computed values,
                     # parsing fixtures (respx-mocked, never live HTTP),
                     # merge precedence, route contracts
uv run ruff check src tests
```

Layout follows the spec: `src/tokenomics/schemas/` (contract), `cost/` (pure
engine), `providers/` (OpenRouter/local + TTL cache), `benchmarks/` (AA +
tier derivation + overrides), `registry/` (merge + queries), `service.py`
(facade), `app.py` (thin routes), `client.py` (typed HTTP client).
