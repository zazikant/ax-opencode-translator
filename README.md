# Ax Translator — DSPy-like Translation Pipeline

Translate text into clean, understandable language using a DSPy-inspired pipeline powered by GLM 5.1 via OpenCode.

**Live App:** [https://ax-opencode-translator.vercel.app](https://ax-opencode-translator.vercel.app)

---

## Quick Start — Curl Commands

### 1. Translate (via deployed app — fast mode)

```bash
curl -X POST "https://ax-opencode-translator.vercel.app/api/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Convert telegraphic notes into a structured, circular dependencies removal. Preserve Facts, headings, subheadings, bullet points. Add Argumentative connectives and logical flow. Style polished.\n\nInput::\n\nmodule A imports B; module B imports C; module C imports A; circular dependency detected; app crashes on startup; refactor needed; A should not depend on C; extract shared logic to D; A imports D; C imports D; B unchanged; test all modules; integration test passes; deploy fix",
    "sourceLanguage": "en",
    "targetLanguage": "en",
    "fast": true
  }'
```

### 2. Translate (full pipeline — Translate → Validate → Refine)

```bash
curl -X POST "https://ax-opencode-translator.vercel.app/api/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text here",
    "sourceLanguage": "en",
    "targetLanguage": "hi",
    "fast": false
  }'
```

### 3. NVIDIA API Direct (openai/gpt-oss-120b — same transformation engine)

```bash
curl -X POST "https://integrate.api.nvidia.com/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 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" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [
      {
        "role": "system",
        "content": "You are an expert text transformation assistant. Follow the users instructions precisely and produce the requested output.\n\nRules:\n- The users text contains transformation instructions — follow them exactly\n- Produce structured output (headings, bullet points, etc.) when requested\n- Expand concepts into their natural sub-components when explaining technical terms\n- Do NOT invent specific named software products unless explicitly in the input\n- Output ONLY the transformed text, nothing else"
      },
      {
        "role": "user",
        "content": "Convert telegraphic notes into a structured, circular dependencies removal. Preserve Facts, headings, subheadings, bullet points. Add Argumentative connectives and logical flow. Style polished.\n\nInput::\n\n<YOUR TELEGRAPHIC NOTES HERE>"
      }
    ],
    "temperature": 0.5,
    "top_p": 1,
    "max_tokens": 4096,
    "stream": false
  }'
```

### 4. Check Server API Key Status

```bash
curl -s "https://ax-opencode-translator.vercel.app/api/config"
```

Returns: `{"hasServerKey": true}`

---

## Example Prompts

### Telegraphic Speech

```
Write a telegraphic speech based on the idea:

scientist lab late night experiment fail again coffee cold notes scattered breakthrough near feel it data wrong somewhere mystery deep focus sharp deadline tomorrow funding cut fear ignore push forward

Answer:

Scientist in the lab, late night; experiment fails again; coffee is cold; notes scattered; breakthrough feels near; data wrong somewhere; mystery deep; focus sharp; deadline tomorrow; funding cut; fear ignored; push forward.
```

### Tag-Based Visual Tokens

```
Use tag-based prompts only. No grammar, no connectives, just core visual/semantic tokens:

couple glasses winter coats selfie snow mountains alpine valley chalets blue sky clouds snowing warm smiles

Answer:

Couple in glasses, winter coats, selfie amid snow-capped mountains, alpine valley, chalets under blue sky, clouds, snowfall, warm smiles.
```

### Circular Dependencies Removal

```
Convert telegraphic notes into a structured, circular dependencies removal. Preserve Facts, headings, subheadings, bullet points. Add Argumentative connectives and logical flow. Style polished.

Input::

module A imports B; module B imports C; module C imports A; circular dependency detected; app crashes on startup; refactor needed; A should not depend on C; extract shared logic to D; A imports D; C imports D; B unchanged; test all modules; integration test passes; deploy fix
```

### Extract Action Items

```
Extract only action items:

The team needs to finish the report, schedule a meeting, and send invoices by Friday

- Finish the report
- Schedule a meeting
- Send invoices by Friday
```

### Compress to Keywords

```
Compress to keywords:

The economy is struggling due to inflation

economy, inflation, struggling
```

---

## Features

- **3-Stage Pipeline**: Translate -> Validate -> Refine (DSPy-inspired)
- **Quality Scoring**: Automatic validation with quality score (0-100)
- **Surgical Refinement**: If quality is low, targeted fixes are applied
- **Same-Language Transformation**: en->en mode for telegraphic notes, circular dependency removal, keyword compression, etc.
- **26 Languages**: Including Hindi, Spanish, French, Japanese, Chinese, Arabic, and more
- **Auto-Detect**: Automatic source language detection
- **Session-Only API Key**: Your OpenCode API key is never stored on the server

## Tech Stack

- **Frontend**: Next.js 16, React 19, shadcn/ui, Tailwind CSS
- **Backend**: Next.js API Routes with embedded pipeline
- **LLM**: GLM 5.1 via OpenCode (`opencode.ai`) -- OpenAI Chat Completions API compatible
- **Alt LLM**: NVIDIA NIM `openai/gpt-oss-120b` -- same OpenAI-compatible interface

## Getting Started

### Prerequisites

- Node.js 18+
- npm or bun
- OpenCode API key from [opencode.ai](https://opencode.ai/)

### Installation

```bash
git clone https://github.com/zazikant/ax-opencode-translator.git
cd ax-opencode-translator
npm install
npm run dev
```

### Environment Variables

Set `OPENCODE_API_KEY` on the server to enable the full pipeline (Translate -> Validate -> Refine) without requiring users to enter a key in the UI.

```bash
# Required for server-side full pipeline mode
OPENCODE_API_KEY=sk-xxxxx
```

If no server-side key is set, the app runs in **fast mode** (translate only) and the user must provide their own API key in the browser UI.

### Deploy to Vercel

1. Push to GitHub
2. Import repo in [vercel.com](https://vercel.com)
3. Set the `OPENCODE_API_KEY` environment variable in Vercel project settings
4. Deploy!

## How the Pipeline Works

### Stage 1: Translate
GLM 5.1 translates your text with a carefully compiled system prompt that preserves meaning and tone.

### Stage 2: Validate
A separate LLM call evaluates accuracy, fluency, completeness, and terminology. Returns a quality score (0-100) and list of issues.

### Stage 3: Refine
If validation finds issues, a surgical fix prompt is compiled (DSPy-style) targeting only the problems. Up to 2 refinements.

## DSPy/Ax Design Principles

- **Signature-based prompt compilation** (`compileTranslatePrompt`)
- **Mode A (initial)**: Empty error history -> fresh translation prompt
- **Mode B (surgical fix)**: Latest error + previous attempts -> focused fix prompt
- **Workflow never passes raw error history** -- only compiled prompts
- **Each step is a discrete, testable activity** (like DSPy Signatures)

## Project Structure

```
src/
  app/
    page.tsx              # Main translation UI
    layout.tsx            # App layout & metadata
    api/translate/
      route.ts            # API route (calls pipeline)
    api/config/
      route.ts            # Config endpoint (hasServerKey check)
  lib/
    llm-client.ts           # GLM 5.1 API client (OpenAI Chat Completions compatible)
    translation-pipeline.ts  # DSPy-like pipeline
  components/ui/            # shadcn/ui components
```

## License

MIT
