# Ax Translator

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
