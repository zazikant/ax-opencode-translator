/**
 * GLM 5.x API Client — OpenAI Chat Completions interface
 *
 * Calls the GLM model via the OpenAI-compatible chat completions endpoint.
 * Uses system + user message format (same as OpenAI SDK).
 *
 * Base URL: https://opencode.ai/zen/go
 * Default model: glm-5.1 (alias — currently routes to GLM-5.3 on the gateway)
 * Endpoint: /v1/chat/completions (OpenAI-compatible)
 * Auth: Authorization: Bearer header
 *
 * GLM 5.3 (released 2026-08-18) is a thinking-only model — reasoning cannot be
 * disabled. We use reasoning_effort: "low" to keep latency/cost low while still
 * producing the final answer in `content`. The parser also reads
 * `reasoning_content` as a safety fallback if the gateway ever returns content
 * there.
 *
 * Session routing:
 * Per https://opencode.ai/docs/go/#where-can-i-use-it, the gateway requires an
 * `x-opencode-session` header for request routing. Enforcement tightened on
 * 2026-09-06; requests without it now return HTTP 400 with
 * `MissingSessionID`. We generate a stable UUID per pipeline run and reuse it
 * across all calls in the same pipeline so the gateway can apply prompt-cache
 * affinity. A fresh UUID is generated if the caller doesn't supply one.
 */

import { randomUUID } from 'node:crypto';

const LLM_BASE_URL = 'https://opencode.ai/zen/go';
const DEFAULT_MODEL = 'glm-5.1';

export interface LLMChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface LLMChatOptions {
  model?: string;
  messages: LLMChatMessage[];
  temperature?: number;
  maxTokens?: number;
  apiKey: string; // Required — always pass explicitly
  sessionId?: string; // Stable per-conversation ID for OpenCode routing/cache affinity
}

export interface LLMChatResponse {
  content: string;
  model: string;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

/**
 * Build the request body for GLM 5.x.
 * reasoning_effort: "low" — GLM 5.3 (currently served behind glm-5.1) is thinking-only
 * and rejects "none". "low" keeps reasoning overhead minimal so most tokens still go
 * to the actual content output. The system prompt in translation-pipeline.ts guides
 * quality.
 */
function buildRequestBody(
  model: string,
  messages: LLMChatMessage[],
  maxTokens: number,
  temperature: number,
): Record<string, unknown> {
  return {
    model,
    messages,
    max_tokens: maxTokens,
    temperature,
    reasoning_effort: 'low',
    stream: false,
  };
}

export async function llmChatCompletion(options: LLMChatOptions): Promise<LLMChatResponse> {
  const model = options.model || DEFAULT_MODEL;
  const sessionId = options.sessionId || randomUUID();
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 50_000); // 50s — leaves 10s buffer for Vercel's 60s maxDuration

  try {
    const response = await fetch(`${LLM_BASE_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${options.apiKey}`,
        'x-opencode-session': sessionId,
      },
      body: JSON.stringify(buildRequestBody(
        model,
        options.messages,
        options.maxTokens ?? 4096,
        options.temperature ?? 0.7,
      )),
      signal: controller.signal,
    });

    clearTimeout(timeout);

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`GLM API error (${response.status}): ${errText}`);
    }

    const data = await response.json();
    const message = data.choices?.[0]?.message;
    // GLM 5.3 returns the final answer in content and reasoning in reasoning_content.
    // Fallback to reasoning_content for safety if content is empty (e.g., token limit hit).
    const content = message?.content || message?.reasoning_content || '';

    if (!content) {
      throw new Error('GLM API returned empty response');
    }

    return {
      content,
      model: data.model || model,
      usage: data.usage,
    };
  } catch (err: unknown) {
    clearTimeout(timeout);
    if (err instanceof Error && err.name === 'AbortError') {
      throw new Error('GLM API request timed out after 50s. The input may be too long for a single request — try shorter text or fewer terms.');
    }
    throw err;
  }
}

/**
 * Convenience: Call with system prompt + user content + API key.
 * This is the primary way the pipeline calls the LLM.
 * Uses reasoning_effort: "low" — GLM 5.3 is thinking-only, so we keep reasoning
 * minimal to preserve latency/cost while still getting the final answer in content.
 */
export async function callLLM(
  systemPrompt: string,
  userContent: string,
  apiKey: string,
  model?: string,
  maxTokens: number = 4096,
  temperature: number = 0.3,
  sessionId?: string
): Promise<string> {
  const modelName = model || DEFAULT_MODEL;
  const resolvedSessionId = sessionId || randomUUID();
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 50_000); // 50s — leaves 10s buffer for Vercel's 60s maxDuration

  try {
    const response = await fetch(`${LLM_BASE_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
        'x-opencode-session': resolvedSessionId,
      },
      body: JSON.stringify(buildRequestBody(
        modelName,
        [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userContent },
        ],
        maxTokens,
        temperature,
      )),
      signal: controller.signal,
    });

    clearTimeout(timeout);

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`GLM API error (${response.status}): ${errText}`);
    }

    const data = await response.json();
    const message = data.choices?.[0]?.message;
    // With reasoning_effort: "low", content should be populated.
    // Fallback to reasoning_content just in case the gateway behaves unexpectedly.
    const content = message?.content || message?.reasoning_content || '';

    if (!content) {
      console.error('[GLM] Empty response. Full data:', JSON.stringify(data).substring(0, 500));
      throw new Error('GLM API returned empty response');
    }
    console.log(`[GLM] Response received. Content length: ${content.length}, preview: "${content.substring(0, 150)}"`);
    return content;
  } catch (err: unknown) {
    clearTimeout(timeout);
    if (err instanceof Error && err.name === 'AbortError') {
      throw new Error('GLM API request timed out after 50s. The input may be too long for a single request — try shorter text or fewer terms.');
    }
    throw err;
  }
}

export { DEFAULT_MODEL, LLM_BASE_URL };
