/**
 * Ax Translation Pipeline — DSPy-like orchestration
 *
 * This is the core pipeline that was originally designed as Temporal workflows.
 * It runs directly inside Next.js for reliability, but preserves all the
 * Temporal-inspired patterns from the Second Brain:
 *
 * - compileTranslatePrompt: Like DSPy's Module.compile() — produces focused prompts
 * - ErrorEntry tracking: Full error history for surgical retries
 * - resumeFrom state machine: Deterministic pipeline progression
 * - Activity-style discrete steps: translate → validate → refine
 *
 * Pipeline flow:
 * 1. Translate text (initial attempt)
 * 2. Validate translation quality
 * 3. If validation fails, refine translation (up to 2 refinements)
 * 4. Return final translated text with metadata
 *
 * Fast mode (for Vercel Hobby / timeout constraints):
 * - Single translate call only, no validate/refine
 */

import { callLLM, DEFAULT_MODEL } from './llm-client';

// ─── Types ───────────────────────────────────────────────────────────────────

export interface TranslationRequest {
  text: string;
  sourceLanguage: string;
  targetLanguage: string;
  apiKey: string;
  model?: string;
}

export interface TranslationResult {
  translatedText: string;
  qualityScore: number;
  attempts: number;
  refinements: number;
  issues?: string[];
  model: string;
  pipeline: string[]; // Which stages ran
}

interface ErrorEntry {
  attempt: number;
  stage: 'translate' | 'validate' | 'refine';
  error: string;
  issues?: string[];
}

// ─── Token Estimation ────────────────────────────────────────────────────────
// Rough: 1 token ≈ 4 chars for English, 2 chars for CJK

function estimateTokens(text: string): number {
  const cjkChars = (text.match(/[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]/g) || []).length;
  const otherChars = text.length - cjkChars;
  return Math.ceil(cjkChars / 2 + otherChars / 4);
}

/**
 * Calculate max_tokens for the output based on input length.
 * GLM 5.1 is a thinking model — it uses tokens for internal reasoning before
 * producing output. We must allocate enough tokens for BOTH reasoning + output.
 *
 * IMPORTANT: On Vercel, each LLM call must complete within 50s (leaving buffer
 * for the 60s maxDuration). Higher max_tokens = longer generation time.
 * We cap at 8192 to keep response times reasonable.
 *
 * For same-language transformation (en→en), output can be 3-5× the input
 * (telegraphic notes → structured essay with explanations).
 * For cross-language translation, output ≈ input × 1.5.
 *
 * Minimum 2048, maximum 8192.
 */
function calculateMaxTokens(inputText: string, isSameLanguage: boolean = false): number {
  const inputTokens = estimateTokens(inputText);
  // Same-language transformation produces much longer output (3-5× expansion)
  const multiplier = isSameLanguage ? 4 : 1.5;
  const outputTokens = Math.ceil(inputTokens * multiplier);
  return Math.max(2048, Math.min(8192, outputTokens));
}

// ─── Echo Detection ─────────────────────────────────────────────────────────
// If the LLM returns the same text it was given (instead of translating),
// we detect it and retry with a more forceful prompt.

function isEcho(originalText: string, translatedText: string): boolean {
  const normalize = (s: string) => s.trim().toLowerCase().replace(/\s+/g, ' ');
  return normalize(originalText) === normalize(translatedText);
}

// ─── compileTranslatePrompt — pure function (DSPy-like) ──────────────────────
// Like DSPy's Module.compile() — produces focused prompts based on error history.
// This is the ONLY place prompt construction happens (same rule as Temporal workflow).

function compileTranslatePrompt(
  input: TranslationRequest,
  errorHistory: ErrorEntry[],
  stage: 'translate' | 'validate' | 'refine'
): string {
  // No errors yet — this is initial context
  if (errorHistory.length === 0) {
    return `Initial translation request from ${input.sourceLanguage} to ${input.targetLanguage}`;
  }

  // Build surgical context from error history (Mode B: surgical fix)
  const latestError = errorHistory[errorHistory.length - 1];
  const previousAttempts = errorHistory.slice(0, -1).map(e =>
    `  Attempt ${e.attempt} | ${e.stage}: ${e.error.substring(0, 200)}`
  ).join('\n');

  return `Refinement context for stage "${stage}":
Latest issue (attempt ${latestError.attempt}, stage ${latestError.stage}): ${latestError.error.substring(0, 300)}
${latestError.issues ? `Issues: ${latestError.issues.join(', ')}` : ''}

Previous attempts — do NOT repeat these patterns:
${previousAttempts || '  None yet.'}

Source: ${input.text.substring(0, 200)}... → ${input.targetLanguage}`;
}

// ─── Activity 1: translateText ───────────────────────────────────────────────

async function translateText(input: TranslationRequest, isRetry: boolean = false): Promise<{ translatedText: string; model: string }> {
  const srcLabel = input.sourceLanguage === 'auto' ? 'the detected source language' : input.sourceLanguage;
  const targetLabel = input.targetLanguage;

  // Detect same-language mode (en→en, hi→hi, etc.) — this is text transformation, not translation
  const isSameLanguage = input.sourceLanguage === input.targetLanguage ||
    (input.sourceLanguage === 'auto' && input.targetLanguage === 'en'); // auto likely = en for this app

  let systemPrompt: string;
  let userContent: string;

  if (isSameLanguage && !isRetry) {
    // ─── Same-Language Text Transformation Mode ─────────────────────────────
    // When source and target are the same language, the user is not "translating"
    // but rather transforming/enhancing text (e.g., converting telegraphic notes
    // into a structured essay, compressing to keywords, extracting action items).
    // The user's input text contains embedded instructions that MUST be followed.
    systemPrompt = `You are an expert text transformation assistant. The user provides text that contains INSTRUCTIONS followed by content. Your job is to follow those instructions precisely and produce the requested output.

CRITICAL RULES:
- Read the user's text carefully. It contains transformation instructions (e.g., "Convert telegraphic notes into a formal essay", "Compress to keywords", "Extract action items", etc.)
- Follow ALL instructions in the user's text exactly — formatting, style, structure, headings, bullet points, connectives, etc.
- When the user asks for headings, subheadings, bullet points, or structured output — YOU MUST produce them. Never output a single flat paragraph when structured output is requested.
- When the user asks for argumentative connectives and logical flow — use them: furthermore, consequently, therefore, however, in contrast, moreover, nevertheless, accordingly.
- When the user asks for explanations of technical terms — provide detailed explanations of what each term means and why it was chosen.
- When the user asks for a polished or formal style — write with professional, academic-quality prose.
- When the user asks for "telegraphic speech" — expand each keyword/phrase into a brief grammatical clause (with articles, verb conjugations) and join them with semicolons into a single flowing sentence. Do NOT use periods or line breaks between clauses. Format: "Subject verb object; verb adjective; verb adverb; action phrase."
- When the user asks to "extract action items" — list each distinct action as a bullet point. Keep deadlines/context attached to their action (e.g., "Send invoices by Friday", NOT "Send invoices" + "Complete by Friday"). Do NOT fabricate or infer actions not stated in the input.
- Preserve ALL facts and information from the input. Do not add fabricated information.
- Output ONLY the transformed text. No preamble, no meta-commentary, no labels like "Here is the output:".`;

    userContent = input.text;
  } else if (isSameLanguage && isRetry) {
    // ─── Same-Language Retry ────────────────────────────────────────────────
    systemPrompt = `You are an expert text transformation assistant. The user provides text that contains INSTRUCTIONS followed by content. Your job is to follow those instructions precisely and produce the requested output.

CRITICAL: The previous attempt produced output that was too similar to the input or did not follow the user's transformation instructions. You MUST actually transform the text according to the instructions.

RULES:
- Read the user's text carefully and follow ALL embedded instructions precisely.
- When the user asks for structured output (headings, bullet points, sections), produce them — never output a flat paragraph.
- When the user asks for connectives and logical flow, use them explicitly.
- When the user asks for explanations of terms, provide them.
- When the user asks for "telegraphic speech" — expand each keyword into a brief grammatical clause with articles and verb conjugations, joined by semicolons into one flowing sentence. No periods or line breaks between clauses.
- Output ONLY the transformed text. No preamble or meta-commentary.`;

    userContent = input.text;
  } else if (isRetry) {
    // ─── Cross-Language Retry ───────────────────────────────────────────────
    // More forceful prompt for retries — explicitly say NOT to echo
    systemPrompt = `You are a professional translator. Your task is to TRANSLATE the given text from ${srcLabel} into ${targetLabel}.

CRITICAL INSTRUCTION: You MUST output the text IN ${targetLabel.toUpperCase()}. Do NOT output the same text in the original language. This is a translation task, not a repetition task. The previous attempt returned the original text unchanged — you must actually translate it this time.

Rules:
- Produce a clean, natural, and understandable translation in ${targetLabel}
- Preserve the original meaning exactly — do not add, remove, or change information
- Use natural phrasing that a native speaker of ${targetLabel} would use
- Output ONLY the translated text in ${targetLabel}, nothing else.`;

    userContent = `Translate the following text from ${srcLabel} to ${targetLabel}. The output must be in ${targetLabel}:\n\n${input.text}`;
  } else {
    // ─── Cross-Language Translation (default, matches ax-translator) ───────
    systemPrompt = `You are a professional translator. Translate the given text from ${srcLabel} to ${targetLabel}.

Rules:
- Produce a clean, natural, and understandable translation
- Preserve the original meaning exactly — do not add, remove, or change information
- Use natural phrasing that a native speaker would use
- Maintain the same tone and register (formal, informal, technical, etc.)
- If the text contains idioms, translate them to equivalent expressions in the target language
- If the text contains technical terms, use the standard terminology in the target language
- Output ONLY the translated text in ${targetLabel}, nothing else
- Do NOT output the original text — you must output the translation`;

    userContent = `Translate the following text from ${srcLabel} to ${targetLabel}. The output must be in ${targetLabel}:\n\n${input.text}`;
  }

  console.log(`[Pipeline] translateText (retry=${isRetry}): src=${srcLabel}, target=${targetLabel}, input length=${input.text.length}`);

  const maxTokens = calculateMaxTokens(input.text, isSameLanguage);
  console.log(`[Pipeline] Dynamic max_tokens: ${maxTokens} (input est. ${estimateTokens(input.text)} tokens)`);

  // Use higher temperature for same-language transformation (more creative/structured output)
  const temperature = isSameLanguage ? 0.5 : 0.3;
  const result = await callLLM(systemPrompt, userContent, input.apiKey, input.model, maxTokens, temperature);

  // Strip any markdown code blocks or quotes the LLM might add
  const cleaned = result
    .replace(/^```[\w]*\n?/m, '')
    .replace(/\n?```$/m, '')
    .replace(/^["']|["']$/g, '')
    .trim();

  console.log(`[Pipeline] translateText result: length=${cleaned.length}, preview="${cleaned.substring(0, 100)}", isEcho=${isEcho(input.text, cleaned)}`);

  return {
    translatedText: cleaned,
    model: input.model || DEFAULT_MODEL,
  };
}

// ─── Activity 2: validateTranslation ─────────────────────────────────────────

async function validateTranslation(
  input: TranslationRequest,
  translatedText: string
): Promise<{ isValid: boolean; qualityScore: number; issues: string[]; suggestion?: string }> {
  const isSameLanguage = input.sourceLanguage === input.targetLanguage ||
    (input.sourceLanguage === 'auto' && input.targetLanguage === 'en');

  const systemPrompt = isSameLanguage
    ? `You are a text transformation quality reviewer. The user provided text with transformation instructions (e.g., convert to essay, compress to keywords, extract items). Evaluate whether the output correctly follows those instructions.

Evaluate on these criteria:
1. Instruction adherence: Did the output follow ALL the user's instructions (format, style, structure)?
2. Structure: Does the output have the requested headings, subheadings, bullet points, or sections? (Flat paragraphs when structured output was requested = FAIL)
3. Completeness: Is any information from the input missing?
4. Quality: Is the output well-written with proper connectives, logical flow, and polish as requested?
5. Depth: Are technical terms explained when requested? Are arguments developed with reasoning?

Respond in this exact JSON format:
{
  "isValid": true/false,
  "qualityScore": 0-100,
  "issues": ["issue1", "issue2"],
  "suggestion": "optional improvement suggestion"
}

Be strict: if the output is a flat paragraph when the user asked for structured output with headings/bullet points, set isValid to false and qualityScore below 50.`
    : `You are a translation quality reviewer. Evaluate the provided translation and respond in JSON format.

Evaluate on these criteria:
1. Accuracy: Does the translation preserve the original meaning?
2. Fluency: Is the translation natural and well-formed in the target language?
3. Completeness: Is any information missing or added?
4. Terminology: Are technical terms translated correctly?

Respond in this exact JSON format:
{
  "isValid": true/false,
  "qualityScore": 0-100,
  "issues": ["issue1", "issue2"],
  "suggestion": "optional improvement suggestion"
}

If the translation is good enough for practical use, set isValid to true even if minor improvements are possible.`;

  const srcLabel = input.sourceLanguage === 'auto' ? 'detected' : input.sourceLanguage;
  const userContent = `Source text (${srcLabel}):
"""
${input.text}
"""

Translation (${input.targetLanguage}):
"""
${translatedText}
"""`;

  // GLM 5.1 is a thinking model — needs at least 2048 max_tokens for reasoning + output
  const result = await callLLM(systemPrompt, userContent, input.apiKey, input.model, 2048, 0.1);

  try {
    const jsonMatch = result.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      return { isValid: true, qualityScore: 70, issues: ['Could not parse validation response'] };
    }
    const parsed = JSON.parse(jsonMatch[0]);
    return {
      isValid: parsed.isValid ?? true,
      qualityScore: parsed.qualityScore ?? 70,
      issues: parsed.issues ?? [],
      suggestion: parsed.suggestion,
    };
  } catch {
    return { isValid: true, qualityScore: 70, issues: ['Could not parse validation response'] };
  }
}

// ─── Activity 3: refineTranslation ───────────────────────────────────────────

async function refineTranslation(
  input: TranslationRequest,
  translatedText: string,
  issues: string[]
): Promise<string> {
  const issuesList = issues.map(i => `- ${i}`).join('\n');
  const isSameLanguage = input.sourceLanguage === input.targetLanguage ||
    (input.sourceLanguage === 'auto' && input.targetLanguage === 'en');

  const systemPrompt = isSameLanguage
    ? `You are an expert text transformation assistant refining output to better match the user's instructions.
Fix ALL the issues identified while preserving what already works.

CRITICAL: If the user asked for structured output (headings, bullet points, sections) and the current output is a flat paragraph, you MUST restructure it with proper headings, subheadings, bullet points, and sections.
If the user asked for argumentative connectives, add them: furthermore, consequently, therefore, however, moreover, nevertheless.
If the user asked for explanations of terms, add detailed explanations.
Output ONLY the improved transformed text, nothing else.`
    : `You are a professional translator refining a translation.
Fix ALL the issues identified while keeping the rest of the translation unchanged.
Output ONLY the improved translation, nothing else.`;

  const srcLabel = input.sourceLanguage === 'auto' ? 'detected' : input.sourceLanguage;
  const userContent = `Source text (${srcLabel}):
"""
${input.text}
"""

Current translation (${input.targetLanguage}):
"""
${translatedText}
"""

Issues found with the current translation:
${issuesList}`;

  const result = await callLLM(systemPrompt, userContent, input.apiKey, input.model, calculateMaxTokens(translatedText, isSameLanguage), 0.2);

  return result
    .replace(/^```[\w]*\n?/m, '')
    .replace(/\n?```$/m, '')
    .replace(/^["']|["']$/g, '')
    .trim();
}

// ─── Fast Translation (single call, no validate/refine) ──────────────────────
// For Vercel Hobby plan or when speed is preferred over quality validation.

export async function runFastTranslation(input: TranslationRequest): Promise<TranslationResult> {
  console.log('[Pipeline] Fast mode: translate only (no validate/refine)');

  let translatedText = '';
  let model = input.model || DEFAULT_MODEL;
  const pipeline: string[] = ['fast-translate'];

  try {
    const result = await translateText(input);
    translatedText = result.translatedText;
    model = result.model;

    // Echo detection — retry once if model echoed input
    if (isEcho(input.text, translatedText) && input.sourceLanguage !== input.targetLanguage) {
      console.log('[Pipeline] Echo detected in fast mode — retrying...');
      pipeline.push('echo-detected', 'fast-retry');
      const retryResult = await translateText(input, true);
      translatedText = retryResult.translatedText;
    }
  } catch (err: unknown) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    console.error('[Pipeline] Fast translation failed:', errorMsg);
    return {
      translatedText: '',
      qualityScore: 0,
      attempts: 1,
      refinements: 0,
      issues: [errorMsg],
      model,
      pipeline,
    };
  }

  return {
    translatedText,
    qualityScore: 85, // Estimated — no validation step
    attempts: 1,
    refinements: 0,
    issues: undefined,
    model,
    pipeline,
  };
}

// ─── Main Pipeline (was translateWorkflow) ───────────────────────────────────
// This is the Temporal workflow logic, running directly.
// State machine: translate → validate → refine → done

export async function runTranslationPipeline(input: TranslationRequest): Promise<TranslationResult> {
  console.log(`[Pipeline] Full pipeline: src=${input.sourceLanguage}, target=${input.targetLanguage}, text length=${input.text.length}`);

  let attempt = 0;
  let refinements = 0;
  const maxRefinements = 2;
  let resumeFrom: 'translate' | 'validate' | 'refine' | 'done' = 'translate';
  const errorHistory: ErrorEntry[] = [];
  const pipeline: string[] = [];

  let translatedText = '';
  let qualityScore = 0;
  let issues: string[] = [];
  let model = input.model || DEFAULT_MODEL;

  // ─── Stage 1: Translate ─────────────────────────────────────────────────

  if (resumeFrom === 'translate') {
    attempt++;
    pipeline.push('translate');

    try {
      const result = await translateText(input);
      translatedText = result.translatedText;
      model = result.model;

      // ── Echo Detection: If model returned the same text, retry with forceful prompt ──
      if (isEcho(input.text, translatedText) && input.sourceLanguage !== input.targetLanguage) {
        console.log('[Pipeline] Echo detected — model returned input text. Retrying with explicit prompt...');
        pipeline.push('echo-detected');
        attempt++;
        pipeline.push('translate-retry');

        const retryResult = await translateText(input, true);
        translatedText = retryResult.translatedText;

        // If still echoing after retry, note it but continue to validation
        if (isEcho(input.text, translatedText)) {
          console.log('[Pipeline] Echo persists after retry — validation will catch this');
          pipeline.push('echo-persist');
        }
      }

      resumeFrom = 'validate';
    } catch (err: unknown) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      errorHistory.push({ attempt, stage: 'translate', error: errorMsg });

      // Retry translation once
      if (attempt < 2) {
        attempt++;
        pipeline.push('translate-retry');

        // Compile fix context (DSPy-like)
        const fixContext = compileTranslatePrompt(input, errorHistory, 'translate');
        console.log(`[Pipeline] Translate retry context: ${fixContext.substring(0, 200)}`);

        try {
          const result = await translateText(input, true);
          translatedText = result.translatedText;
          model = result.model;

          // Echo detection on retry too
          if (isEcho(input.text, translatedText) && input.sourceLanguage !== input.targetLanguage) {
            console.log('[Pipeline] Echo detected on retry — continuing');
            pipeline.push('echo-on-retry');
          }

          resumeFrom = 'validate';
        } catch (err2: unknown) {
          const errorMsg2 = err2 instanceof Error ? err2.message : String(err2);
          errorHistory.push({ attempt, stage: 'translate', error: errorMsg2 });
          return {
            translatedText: '',
            qualityScore: 0,
            attempts: attempt,
            refinements: 0,
            issues: ['Translation failed after retry'],
            model,
            pipeline,
          };
        }
      } else {
        return {
          translatedText: '',
          qualityScore: 0,
          attempts: attempt,
          refinements: 0,
          issues: ['Translation failed'],
          model,
          pipeline,
        };
      }
    }
  }

  // ─── Stage 2: Validate ──────────────────────────────────────────────────

  if (resumeFrom === 'validate') {
    pipeline.push('validate');

    try {
      const validation = await validateTranslation(input, translatedText);
      qualityScore = validation.qualityScore;
      issues = validation.issues;

      // Check if validation caught an echo (translated text same as source)
      if (isEcho(input.text, translatedText) && input.sourceLanguage !== input.targetLanguage) {
        // Force refinement to fix the echo
        pipeline.push('echo-caught-by-validation');
        qualityScore = Math.min(qualityScore, 30);
        issues = [...issues, 'Translation appears identical to source text — not actually translated'];
        resumeFrom = 'refine';
      } else if (validation.isValid) {
        pipeline.push('validate-pass');
        resumeFrom = 'done';
      } else {
        pipeline.push('validate-fail');
        resumeFrom = 'refine';
      }
    } catch (err: unknown) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      errorHistory.push({ attempt, stage: 'validate', error: errorMsg });
      qualityScore = 60;
      resumeFrom = 'done';
    }
  }

  // ─── Stage 3: Refine (up to maxRefinements) ─────────────────────────────

  while (resumeFrom === 'refine' && refinements < maxRefinements) {
    refinements++;
    attempt++;
    pipeline.push(`refine-${refinements}`);

    // Compile surgical fix prompt (DSPy-like focused prompt)
    const fixContext = compileTranslatePrompt(input, errorHistory, 'refine');
    console.log(`[Pipeline] Refinement #${refinements} context: ${fixContext.substring(0, 200)}`);

    try {
      translatedText = await refineTranslation(input, translatedText, issues);

      // Re-validate after refinement
      pipeline.push(`revalidate-${refinements}`);
      try {
        const revalidation = await validateTranslation(input, translatedText);
        qualityScore = revalidation.qualityScore;
        issues = revalidation.issues;

        if (revalidation.isValid && !isEcho(input.text, translatedText)) {
          pipeline.push(`revalidate-pass-${refinements}`);
          resumeFrom = 'done';
          break;
        } else {
          pipeline.push(`revalidate-fail-${refinements}`);
          if (refinements >= maxRefinements) {
            resumeFrom = 'done';
            break;
          }
        }
      } catch (err: unknown) {
        const errorMsg = err instanceof Error ? err.message : String(err);
        errorHistory.push({ attempt, stage: 'validate', error: errorMsg });
        qualityScore = 65;
        resumeFrom = 'done';
        break;
      }
    } catch (err: unknown) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      errorHistory.push({ attempt, stage: 'refine', error: errorMsg, issues });

      if (refinements >= maxRefinements) {
        resumeFrom = 'done';
        break;
      }
    }
  }

  return {
    translatedText,
    qualityScore,
    attempts: attempt,
    refinements,
    issues: issues.length > 0 ? issues : undefined,
    model,
    pipeline,
  };
}
