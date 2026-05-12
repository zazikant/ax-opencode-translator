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
 * GLM 5.1 is a thinking/reasoning model. The opencode.ai gateway does not
 * support disabling thinking mode — enable_thinking, chat_template_kwargs,
 * and reasoning_effort (int) are all rejected as extra inputs. Thinking
 * is always on. We allocate enough max_tokens for reasoning + output
 * and fall back to reasoning_content when content is empty.
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
 * Build the request body for GLM 5.1.
 * No thinking-disabling params — the opencode.ai gateway rejects them all.
 * GLM 5.1 always thinks; we allocate enough tokens for reasoning + output.
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
    const message = data.choices?.[0]?.message;
    // GLM 5.1 returns reasoning in reasoning_content and the final answer in content.
    // If content is empty (token limit hit during reasoning), fall back to reasoning_content.
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
    const message = data.choices?.[0]?.message;
    // GLM 5.1 returns reasoning in reasoning_content and the final answer in content.
    // If content is empty (token limit hit during reasoning), fall back to reasoning_content.
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
