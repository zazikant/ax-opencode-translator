# Ax Translator — Curl Commands

### 1. Translate (deployed app — fast mode)

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

### 3. NVIDIA API Direct (openai/gpt-oss-120b)

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
