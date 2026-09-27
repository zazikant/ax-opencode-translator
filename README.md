# Ax Translator

Test with example to solve a complex thinking 

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
