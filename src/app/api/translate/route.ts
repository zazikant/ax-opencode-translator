import { NextRequest, NextResponse } from 'next/server';

// Vercel serverless function timeout — translation pipeline can take up to 60s
// (Requires Vercel Pro. On Hobby plan, max is 10s which may timeout on long texts)
export const maxDuration = 60;

import { runTranslationPipeline, runFastTranslation } from '@/lib/translation-pipeline';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { text, sourceLanguage, targetLanguage, apiKey, model, fast } = body;

    // API key: use the one from frontend, or fall back to env variable
    const resolvedApiKey = apiKey || process.env.OPENCODE_API_KEY;

    if (!text || !targetLanguage || !resolvedApiKey) {
      return NextResponse.json(
        { error: 'Missing required fields: text, targetLanguage, apiKey (or set OPENCODE_API_KEY env var)' },
        { status: 400 }
      );
    }

    const input = {
      text,
      sourceLanguage: sourceLanguage || 'auto',
      targetLanguage,
      apiKey: resolvedApiKey,
      model: model || undefined,
    };

    // ─── Pipeline Mode Selection ───────────────────────────────────────────
    // GLM 5.1 is a thinking model — each LLM call takes 15-30s.
    // Full pipeline (translate → validate → refine) = 2-5 sequential calls
    // = 60-150s total, which exceeds Vercel's 60s maxDuration.
    //
    // Strategy:
    // - fast=false (explicit) → full pipeline (for hosts with longer timeouts)
    // - Default (fast undefined or fast=true) → fast mode (single translate call)
    //
    // Fast mode produces excellent quality because the same-language
    // transformation prompts are rich and guide GLM 5.1 well.
    const useFastMode = fast !== false;

    const result = useFastMode
      ? await runFastTranslation(input)
      : await runTranslationPipeline(input);

    // If pipeline returned empty text, return error status instead of 200
    if (!result.translatedText || result.translatedText.trim() === '') {
      const errorMsg = result.issues?.[0] || 'Translation returned empty result';
      console.error('[Translate API] Empty result:', errorMsg);
      return NextResponse.json(
        { error: errorMsg },
        { status: 500 }
      );
    }

    return NextResponse.json(result);
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : 'Internal server error';
    console.error('[Translate API] Error:', message);

    // If timeout error, suggest fast mode
    if (message.includes('timeout') || message.includes('timed out')) {
      return NextResponse.json(
        { error: 'Translation timed out. Try using fast mode (add "fast": true to request) or use shorter text.' },
        { status: 504 }
      );
    }

    return NextResponse.json(
      { error: message },
      { status: 500 }
    );
  }
}
