## Write a telegraphic speech based on the idea::

scientist lab late night experiment fail again coffee cold notes scattered breakthrough near feel it data wrong somewhere mystery deep focus sharp deadline tomorrow funding cut fear ignore push forward

Answer:

Scientist in the lab, late night; experiment fails again; coffee is cold; notes scattered; breakthrough feels near; data wrong somewhere; mystery deep; focus sharp; deadline tomorrow; funding cut; fear ignored; push forward.


## Use tag-based prompts only. No grammar, no connectives, just core visual/semantic tokens::


Write a telegraphic speech based on the idea:

couple glasses winter coats selfie snow mountains alpine valley chalets blue sky clouds snowing warm smiles

Answer:

Couple in glasses, winter coats, selfie amid snow‑capped mountains, alpine valley, chalets under blue sky, clouds, snowfall, warm smiles.

## Extract only action items::

The team needs to finish the report, schedule a meeting, and send invoices by Friday

- Finish the report  
- Schedule a meeting  
- Send invoices by Friday

## Write sentence from Telegraphic speech idea::

A man apple hungry eats 

Answer:

A hungry man eats an apple.

## Compress to keywords:

The economy is struggling due to inflation

economy, inflation, struggling

## Convert telegraphic notes into a formal essay.  Preserve Facts, headings, subheadings, bullet points.  Add Argumentative connectives and logical flow. Style Formal, academic, polished.Input::

this is standalone prompt not to be combined with any above 

## Write a telegraphic speech about:: 

Climate change

Rising temperatures; melting ice caps; extreme weather spikes.  
CO₂ emissions soaring; fossil fuels dominate; oceans acidify.  
Deforestation accelerates; biodiversity collapses; sea levels climb.  
Urgent action required: slash emissions, shift to renewables, reforest, enforce climate policies.  
Future of humanity hangs in the balance.

## Write a comedy telegraphic speech about::

Man going to restaurant

Man: starving. Enters restaurant. Waiter: menu. Man: eyes menu, brain: “food?” Orders steak, medium‑rare. Waiter: “how you want it?” Man: “like my ex—well done.” Plate arrives. Man: “is this a joke?” Fork: “no, it’s me.” Soup: “I’m too hot for this.” Bill arrives. Man: “pay with jokes?” Cashier: “only laughs accepted.” Man leaves, satisfied, stomach full, ego slightly bruised.

# Ax Translator — DSPy-like Translation Pipeline

Translate text into clean, understandable language using a DSPy-inspired pipeline powered by GLM 5.1 via OpenCode.

## Features

- **3-Stage Pipeline**: Translate → Validate → Refine (DSPy-inspired)
- **Quality Scoring**: Automatic validation with quality score (0-100)
- **Surgical Refinement**: If quality is low, targeted fixes are applied
- **26 Languages**: Including Hindi, Spanish, French, Japanese, Chinese, Arabic, and more
- **Auto-Detect**: Automatic source language detection
- **Session-Only API Key**: Your OpenCode API key is never stored on the server

## Tech Stack

- **Frontend**: Next.js 16, React 19, shadcn/ui, Tailwind CSS
- **Backend**: Next.js API Routes with embedded pipeline
- **LLM**: GLM 5.1 via OpenCode (`opencode.ai`) — OpenAI Chat Completions API compatible

## Getting Started

### Prerequisites

- Node.js 18+
- npm or bun
- OpenCode API key from [opencode.ai](https://opencode.ai/)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/ax-translator.git
cd ax-translator
npm install
npm run dev
```

### Environment Variables

Set `OPENCODE_API_KEY` on the server to enable the full pipeline (Translate → Validate → Refine) without requiring users to enter a key in the UI.

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
- **Mode A (initial)**: Empty error history → fresh translation prompt
- **Mode B (surgical fix)**: Latest error + previous attempts → focused fix prompt
- **Workflow never passes raw error history** — only compiled prompts
- **Each step is a discrete, testable activity** (like DSPy Signatures)

## Project Structure

```
src/
├── app/
│   ├── page.tsx              # Main translation UI
│   ├── layout.tsx            # App layout & metadata
│   └── api/translate/
│       └── route.ts          # API route (calls pipeline)
├── lib/
│   ├── llm-client.ts           # GLM 5.1 API client (OpenAI Chat Completions compatible)
│   └── translation-pipeline.ts  # DSPy-like pipeline
└── components/ui/            # shadcn/ui components
```

## License

MIT
