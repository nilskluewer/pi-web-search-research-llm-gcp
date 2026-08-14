# pi-web-search-research-llm-gcp

Web search & research tools for the [Pi coding agent](https://pi.dev/), backed by **Vertex AI Gemini** with Google Search grounding and Google Cloud Agent Platform credentials.

This is a Vertex AI variant of the Gemini search extension. It does **not** use a Gemini API key.

## Tools

Registers three tools the model can call:

- **`web_search`** — quick fact-check / verification. Returns a concise synthesized answer + source URLs.
- **`web_research`** — in-depth research on complex topics. Returns a structured answer with documentation details, code snippets, and source URLs.
- **`web_fetch`** — reads one public webpage through Gemini `url_context` and extracts its substantive content into Markdown.

The search and research tools:

- Ground answers with Google Search.
- Can read specific URLs passed in the query via Gemini `url_context`.
- Use high thinking for the configured Gemini 3.5 Flash-Lite and Gemini 3.7 Flash models.
- Resolve Gemini `vertexaisearch.cloud.google.com` redirect links to clean destination URLs.
- Render source URLs as OSC-8 terminal hyperlinks in the Pi TUI when supported by your terminal.
- Prefix results with a compact, brutalist high-contrast Vertex AI provenance/cost summary in Pi TUI, including the actual `flex pricing` or `standard pricing` tier used for that call.

`web_fetch` uses URL Context only, without Google Search. It asks Gemini to:

```text
Extract all information from this webpage into Markdown.
```

The result is model-generated Markdown, not a byte-for-byte copy of the original webpage.

## Vertex AI authentication

Authentication order:

1. `VERTEX_ACCESS_TOKEN` environment variable, if set
2. `gcloud auth print-access-token`

Run this first if needed:

```bash
gcloud auth login
```

## Configuration

Required:

- `VERTEX_PROJECT_ID` — your Google Cloud project ID with Vertex AI enabled.

Defaults:

- `VERTEX_REGION=eu`
- `GEMINI_MODEL_SHORT=gemini-3.5-flash-lite` — model for `web_search`
- `GEMINI_MODEL_LONG=gemini-3.7-flash` — model for `web_research`
- `GEMINI_MODEL_FETCH=gemini-3.1-flash-lite` — model for `web_fetch`

`web_search` and `web_research` use Vertex Gemini thinking level `HIGH`. `web_fetch` uses the lower-cost fetch model without an explicit thinking level.

Set or override these environment variables before starting `pi`. The model IDs must be available in the configured Vertex region.

## Cost estimate configuration

Each result includes a compact estimated cost summary, for example:

```text
◆ Gemini Search [gemini-3.5-flash-lite, eu, €0.0131/$0.0141, 2 sources, standard pricing]
```

Defaults:

- `GEMINI_SEARCH_GROUNDING_USD_PER_1000=14`
- `USD_TO_EUR=0.93`
- `VERTEX_FLEX_TOKEN_DISCOUNT=0.5`

The estimate uses response token counts from Vertex AI and approximate list token rates. Gemini 3.7 Flash research calls use the introductory rate of $0.75 per 1M input tokens and $3.75 per 1M output tokens through December 31, 2026. Regional endpoint pricing may differ. URL Context results report token costs, but the estimate does not add a Google Search grounding charge because `web_fetch` does not use Google Search.
Flex is preferred by default via the Vertex AI Flex PayGo headers.
If the project or region does not support Flex, the extension falls back to the standard tier and reports `standard pricing` in that result.

## Pricing-tier command

Use the Pi command below to inspect or change the pricing preference for the current session:

```text
/search-pricing
/search-pricing flex
/search-pricing standard
```

`flex` is the default and falls back to standard when Flex is unavailable.
`standard` bypasses Flex entirely.
The command does not persist across Pi sessions.

## Install

From npm:

```bash
pi install npm:pi-web-search-research-llm-gcp
```

From a local checkout:

```bash
pi install /path/to/pi-web-search-research-llm-gcp
```

## License

MIT
