# Observability & Evaluation Instrumentation

Knowing that your RAG pipeline *runs* is not the same as knowing it *works well*. Observability gives you traces of every retrieve → prompt → generate step, latency breakdowns, and automated evaluation scores so you can debug regressions before users notice them.

## What you'll learn

- What RAG observability covers and why it differs from standard APM
- Spinning up Arize Phoenix locally for tracing and LLM-based evals
- Running the RAG Triad with TruLens feedback functions
- Writing DeepEval metric tests in Pytest for CI/CD
- A note on OpenLLMetry for framework auto-instrumentation

---

## Why observability is not optional

A RAG system has three distinct failure modes that logs alone cannot surface:

| Failure | Symptom | Requires |
|---|---|---|
| Bad retrieval | Correct answer not in context | Retrieval eval (context relevance) |
| Hallucination | Answer not grounded in context | Groundedness eval |
| Off-topic answer | Answer ignores the question | Answer relevance eval |

Distributed tracing connects these stages into a single timeline so you can pinpoint *which* stage degraded.

---

## Arize Phoenix — local tracing and evaluation

[Arize Phoenix](https://docs.arize.com/phoenix) provides OpenTelemetry-based tracing, LLM-based retrieval and response evaluations, dataset/experiment management, and a local UI — all self-hostable.

!!! warning "License"
    Phoenix is released under the **Elastic License 2.0**. This is a source-available licence but is **not** OSI-approved open source. Review the terms before embedding it in a redistributed product.

### Install and launch

```bash
pip install arize-phoenix
python -m phoenix.server.main serve
```

The UI is available at `http://localhost:6006` by default.

### Auto-instrument your pipeline

Phoenix ships OpenTelemetry instrumentors for LangChain, LlamaIndex, and raw OpenAI calls. Add these lines at the top of your entry-point before any framework imports:

```python
import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# Point traces at the local Phoenix server
provider = TracerProvider()
provider.add_span_processor(
    SimpleSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:6006/v1/traces"))
)
trace.set_tracer_provider(provider)

# Auto-instrument LangChain (swap for LlamaIndexInstrumentor if needed)
LangChainInstrumentor().instrument()
```

Now every LangChain `invoke()` call — including retriever calls and LLM calls — appears as a structured trace in Phoenix with token counts and latencies.

### Running LLM-based evals inside Phoenix

```python
from phoenix.evals import (
    HallucinationEvaluator,
    QAEvaluator,
    RelevanceEvaluator,
    run_evals,
)
from phoenix.evals import OpenAIModel   # swap for a local model wrapper

eval_model = OpenAIModel(model="gpt-4o-mini")  # or any compatible endpoint

# `ds` is a phoenix Dataset exported from the traces UI
results = run_evals(
    dataframe=ds.to_pandas(),
    evaluators=[
        HallucinationEvaluator(eval_model),
        QAEvaluator(eval_model),
        RelevanceEvaluator(eval_model),
    ],
    provide_explanation=True,
)
```

Results are written back into Phoenix and visible per-trace in the UI.

---

## TruLens — the RAG Triad

[TruLens](https://www.trulens.org/) implements the **RAG Triad** as composable feedback functions:

| Metric | Question asked |
|---|---|
| **Context relevance** | Is the retrieved chunk relevant to the question? |
| **Groundedness** | Is the answer supported by the retrieved context? |
| **Answer relevance** | Does the answer actually address the question? |

```python
from trulens.core import TruSession
from trulens.apps.langchain import TruChain
from trulens.providers.openai import OpenAI as TruOpenAI

session = TruSession()
session.reset_database()

provider = TruOpenAI(model_engine="gpt-4o-mini")

f_context_relevance = (
    provider.context_relevance_with_cot_reasons
)
f_groundedness = (
    provider.groundedness_measure_with_cot_reasons
)
f_answer_relevance = (
    provider.relevance_with_cot_reasons
)

# Wrap your existing LangChain chain (from the local-rag project or tutorials)
tru_chain = TruChain(
    chain,                      # your RetrievalQA or LCEL chain
    app_name="local-rag",
    app_version="v1",
    feedbacks=[f_context_relevance, f_groundedness, f_answer_relevance],
)

with tru_chain as recording:
    response = chain.invoke({"query": "What is retrieval-augmented generation?"})

session.get_leaderboard()
```

TruLens also provides OpenTelemetry tracing for agent monitoring, so spans integrate with Phoenix or any OTLP backend.

---

## DeepEval — metrics in Pytest

[DeepEval](https://docs.confident-ai.com/) ships 50+ metrics covering RAG, agents, multi-turn conversations, and safety. Its native Pytest integration means you can gate a merge on eval scores.

```bash
pip install deepeval
```

```python
# test_rag_quality.py
import pytest
from deepeval import assert_test
from deepeval.metrics import (
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
)
from deepeval.test_case import LLMTestCase

@pytest.mark.parametrize("test_case", [
    LLMTestCase(
        input="What chunking strategy works best for PDFs?",
        actual_output="Recursive character splitting with a 512-token chunk size works well for PDFs.",
        retrieval_context=[
            "Recursive character splitting respects paragraph and sentence boundaries...",
            "For PDF documents, a chunk size of 256–512 tokens is commonly recommended...",
        ],
    ),
])
def test_rag_faithfulness(test_case):
    metric = FaithfulnessMetric(threshold=0.8, model="gpt-4o-mini")
    assert_test(test_case, [metric])
```

Run the suite with:

```bash
deepeval test run test_rag_quality.py
```

CI fails if any metric score drops below its threshold, giving you a regression gate without manual inspection.

---

## OpenLLMetry — framework auto-instrumentation

[OpenLLMetry](https://github.com/traceloop/openllmetry) (Apache-2.0) is an OpenTelemetry-based library that auto-instruments 40+ LLM frameworks and providers (LangChain, LlamaIndex, OpenAI, Anthropic, Cohere, and more) with a single `traceloop.init()` call. Traces export to any OTLP backend — including a self-hosted Phoenix or Jaeger instance.

```bash
pip install traceloop-sdk
```

```python
from traceloop.sdk import Traceloop

Traceloop.init(
    app_name="local-rag",
    # disable_batch=True for synchronous local dev
)
```

---

## Connecting observability to evaluation experiments

The recommended workflow for the [local-rag project](../projects/local-rag.md):

1. Instrument with Phoenix or OpenLLMetry during development.
2. Export a trace dataset from the UI after a test run.
3. Run TruLens or DeepEval feedback functions over the dataset.
4. Log scores back to Phoenix experiments to track changes across versions.

This closes the loop between runtime behaviour and offline evaluation — both covered in depth on the [evaluation page](evaluation.md).

---

## Next steps

- Read [Evaluation](evaluation.md) for offline metric design and dataset curation.
- See [Production](production.md) for alerting, dashboards, and cost monitoring at scale.
- The [local-rag project](../projects/local-rag.md) has a working LangChain chain ready to instrument.
