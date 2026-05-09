/**
 * GLM 5.1 API Client — OpenAI Chat Completions interface
 *
 * Calls the GLM 5.1 model via the OpenAI-compatible chat completions endpoint.
 * Uses system + user message format (same as OpenAI SDK).
 *
 * Base URL: https://opencode.ai/zen/go
 * Default model: glm-5.1
 * Endpoint: /v1/chat/completions (OpenAI-compatible)
 * Auth: Authorization: Bearer header
 *
 * THINKING MODE DISABLED: We pass enable_thinking=false and reasoning_effort=0
 * to skip GLM 5.1's internal reasoning tokens. This makes responses 2-3×
 * faster (10-30s instead of 30-90s) which is critical for Vercel's 60s timeout.
 * Output quality remains excellent with well-crafted system prompts.
 */

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
 * Build the request body with thinking disabled.
 * GLM 5.1 supports enable_thinking=false + reasoning_effort=0 to skip
 * internal reasoning tokens, making responses 2-3× faster.
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
    stream: false,
    // Disable thinking mode — saves 50-70% of token budget and response time
    enable_thinking: false,
    reasoning_effort: 0,
  };
}

export async function llmChatCompletion(options: LLMChatOptions): Promise<LLMChatResponse> {
  const model = options.model || DEFAULT_MODEL;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 50_000); // 50s — leaves 10s buffer for Vercel's 60s maxDuration

  try {
    const response = await fetch(`${LLM_BASE_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${options.apiKey}`,
      },
      body: JSON.stringify(buildRequestBody(
        model,
        options.messages,
        options.maxTokens ?? 2048,
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
    const content = data.choices?.[0]?.message?.content || '';

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
 * Thinking is disabled for fast response times on Vercel.
 */
export async function callLLM(
  systemPrompt: string,
  userContent: string,
  apiKey: string,
  model?: string,
  maxTokens: number = 2048,
  temperature: number = 0.3
): Promise<string> {
  const modelName = model || DEFAULT_MODEL;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 50_000); // 50s — leaves 10s buffer for Vercel's 60s maxDuration

  try {
    const response = await fetch(`${LLM_BASE_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
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
    const content = data.choices?.[0]?.message?.content || '';

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
