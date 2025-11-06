# QueryForge Orchestrator (Phase 1)

**QueryForge** provides an **AI-driven deterministic SQL query generation** pipeline, converting **natural language** into a valid **SQL query** through a series of steps:

1. **Natural Language (NL) Input** -> Routers determine the core SQL structure.
2. **Policy Gate** -> Ensures all query components conform to safety rules.
3. **SQL AST Generation** -> Transforms the request into a structured SQL query.

## Features (Phase 1)

- **Single-table SELECT only**
- **Automatic LIMIT of 100** (unless otherwise specified)
- **WHERE filters** based on **status, geo, text**, and **numeric** comparisons
- **Deterministic ranking** with `top`, `bottom`, and `limit`
- **Flexible `ORDER BY`** and **`GROUP BY`** handling
- **Supports "in <place>" geo mapping** for states/cities (US-centric)
- **Query normalization** for date/time (e.g., "this month", "last week")

## CLI Usage

1. **Run Query via Python**:

```python
from orchestrator.orchestrator import orchestrate

# Example queries
print(orchestrate("list advertisers where name contains shop limit 5"))
print(orchestrate("top 5 advertisers order by region"))
