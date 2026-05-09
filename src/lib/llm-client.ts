/**
 * GLM 5.1 API Client — OpenAI Chat Completions interface
 *
 * Calls the GLM 5.1 model via the OpenAI-compatible chat completions endpoint.
 * Uses system + user message format (OpenAI Chat Completions API).
 *
 * Base URL: https://opencode.ai/zen/go
 * Default model: glm-5.1
 * Endpoint: /v1/chat/completions (OpenAI-compatible)
 * Auth: Authorization: Bearer header
 *
 * Note: GLM 5.1 is a thinking model — it uses reasoning_content internally
 * before producing the visible content. We allocate generous max_tokens
 * to ensure the actual response has room after the thinking phase.
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
 * Call GLM 5.1's OpenAI-compatible chat completions API.
 * API key is always passed via options.apiKey as Bearer token.
 */
export async function llmChatCompletion(options: LLMChatOptions): Promise<LLMChatResponse> {
  const model = options.model || DEFAULT_MODEL;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120_000); // 2 min timeout

  try {
    const response = await fetch(`${LLM_BASE_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${options.apiKey}`,
      },
      body: JSON.stringify({
        model,
        messages: options.messages,
        max_tokens: options.maxTokens ?? 4096,
        temperature: options.temperature ?? 0.7,
      }),
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
      throw new Error('GLM API request timed out after 120s');
    }
    throw err;
  }
}

/**
 * Convenience: Call with system prompt + user content + API key.
 * This is the primary way the pipeline calls the LLM.
 *
 * GLM 5.1 is a thinking model — it uses reasoning tokens before the visible
 * response. We use a minimum of 4096 max_tokens to ensure the actual text
 * output has room after the internal reasoning phase.
 */
export async function callLLM(
  systemPrompt: string,
  userContent: string,
  apiKey: string,
  model?: string,
  maxTokens: number = 4096,
  temperature: number = 0.3
): Promise<string> {
  const modelName = model || DEFAULT_MODEL;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120_000);

  try {
    const response = await fetch(`${LLM_BASE_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: modelName,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userContent },
        ],
        max_tokens: maxTokens,
        temperature,
      }),
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
      throw new Error('GLM API request timed out after 120s');
    }
    throw err;
  }
}

export { DEFAULT_MODEL, LLM_BASE_URL };
