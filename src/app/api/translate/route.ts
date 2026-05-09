import { NextRequest, NextResponse } from 'next/server';

// Vercel serverless function timeout — translation pipeline can take up to 60s
// (Requires Vercel Pro. On Hobby plan, max is 10s which may timeout on long texts)
export const maxDuration = 60;

import { runTranslationPipeline, runFastTranslation } from '@/lib/translation-pipeline';

// ─── Token Estimation (mirrors frontend) ──────────────────────────────────────
// Rough: 1 token ≈ 4 chars for English, 2 chars for CJK

function estimateTokens(text: string): number {
  const cjkChars = (text.match(/[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]/g) || []).length;
  const otherChars = text.length - cjkChars;
  return Math.ceil(cjkChars / 2 + otherChars / 4);
}

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
    // Full pipeline (translate → validate → refine) = 2-5 sequential calls.
    // Vercel Pro maxDuration = 60s, so we must be smart about which mode to use.
    //
    // Strategy:
    // 1. fast=true (explicit) → always use fast mode
    // 2. fast=false (explicit) → always use full pipeline
    // 3. Default (fast undefined):
    //    - Large input (>1500 tokens) → fast mode (single call fits in 60s)
    //    - Small input → full pipeline (quality validation worth the time)
    //    - No env key (likely Vercel Hobby) → fast mode

    const inputTokens = estimateTokens(text);
    const LARGE_INPUT_THRESHOLD = 1500; // tokens — above this, full pipeline will likely timeout

    let useFastMode: boolean;
    if (fast === true) {
      useFastMode = true;
    } else if (fast === false) {
      useFastMode = false;
    } else {
      // Auto-decide: fast for large inputs, full pipeline for small
      useFastMode = inputTokens > LARGE_INPUT_THRESHOLD || !process.env.OPENCODE_API_KEY;
    }

    if (useFastMode) {
      console.log(`[Translate API] FAST mode (input: ~${inputTokens} tokens, threshold: ${LARGE_INPUT_THRESHOLD})`);
      const result = await runFastTranslation(input);
      return NextResponse.json(result);
    }

    // Full pipeline: translate → validate → refine
    console.log(`[Translate API] FULL pipeline (input: ~${inputTokens} tokens)`);
    const result = await runTranslationPipeline(input);
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
