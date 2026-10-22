# Ax Translator

Test with example to solve a complex thinking 

## AX DSPy Pipeline

The translation pipeline is now aligned word-to-word with the [atomic-graph-opencode](https://github.com/zazikant/atomic-graph-opencode) AX DSPy reference implementation, while serving this repo's output contract (`translatedText` + `qualityScore` 0-100):

```
repeat up to 3 iterations:
  1. TRANSLATE → translated text   (stage temperature 0.3, dynamic max_tokens)
  2. VALIDATE  → score, issues[]   (stage temperature 0.1, 0.0-1.0 rubric)
  if score >= 0.75:
    break
  else:
  3. REFINE    → corrected text    (stage temperature 0.2, DSPy-compiled fix context)
    go back to step 2
```

Ported patterns:

- **Stage-specific system prompts** — `TRANSLATE_SYSTEM_PROMPT`, `TRANSLATE_RETRY_SYSTEM_PROMPT`, `CROSS_LANGUAGE_SYSTEM_PROMPT`, `CROSS_LANGUAGE_RETRY_SYSTEM_PROMPT`, `VALIDATE_SYSTEM_PROMPT`, `REFINE_SYSTEM_PROMPT` (mirrors the Bible repo's EXTRACT / EXTRACT_RETRY / VALIDATE / REFINE prompts with their CRITICAL OUTPUT RULES blocks)
- **compileRefinePrompt()** — DSPy-style Module.compile(): surgical fix context compiled from the full error history, not generic retries
- **ErrorEntry tracking** — every failed attempt is recorded (attempt, stage, error, issues) and fed into the compiled refinement prompt
- **resumeFrom state machine** — deterministic translate → validate → refine progression, results carried forward between stages
- **Echo detection** — >80% input overlap detection with forceful retry prompt (cross-language mode)
- **Validation threshold** — score >= 0.75 accepted (0.0-1.0 scale, converted to 0-100 at the API boundary)
- **Robust JSON parsing** — parseLLMJson with balanced-bracket extraction and truncation recovery for validation responses
- **Retry logic** — up to 3 retries on 429/5xx with 15-second delays (ported from atomic-graph-opencode's proxy route into the LLM client)
- **Stage-specific temperatures** — translate=0.3, validate=0.1, refine=0.2

## Prompt Examples

The web UI ships with several prebuilt prompt examples, including:

- **Ad-Strategy Landscape Mapper** — exhaustive audience + pillar mining for any product/service
- **Paid-Media Segmentation Strategist** — atomic pillar / audience matrix for ad campaigns
- **Character Pose Stealing** — structured visual mechanics for image prompts
- **Google Flow Image-to-Video Director** — meta-prompt that turns a story idea into 15 stitchable 8-second image-to-video prompts (120s total)
- **Start and End Frame based output** — meta-prompt that produces 3 stitchable 8-second image-to-video clips (24s total) chained via explicit start/end frames
- **Plain English** / **Grammar Fix** — same-language text transformation

## API

```bash
curl -X POST "https://ax-opencode-translator.vercel.app/api/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Convert telegraphic notes into a structured, circular dependicies removal. Preserve Facts, headings, subheadings, bullet points. Add Argumentative connectives and logical flow. Style polished.\n\nInput:\n\n<YOUR NOTES HERE>",
    "sourceLanguage": "en",
    "targetLanguage": "en",
    "fast": true
  }'
```
