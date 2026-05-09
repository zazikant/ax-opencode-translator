/**
 * MiniMax M2.7 API Client — Anthropic Messages-compatible interface
 *
 * Calls MiniMax's M2.7 model via the Anthropic-compatible messages endpoint.
 * Uses system + user message format (Anthropic Messages API).
 *
 * Base URL: https://opencode.ai/zen/go
 * Default model: minimax-m2.7
 * Endpoint: /v1/messages (Anthropic-compatible)
 * Auth: x-api-key header
 */

const MINIMAX_BASE_URL = 'https://opencode.ai/zen/go';
const DEFAULT_MODEL = 'minimax-m2.7';

export interface MinimaxChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface MinimaxChatOptions {
  model?: string;
  messages: MinimaxChatMessage[];
  temperature?: number;
  maxTokens?: number;
  apiKey: string; // Required — always pass explicitly
}

export interface MinimaxChatResponse {
  content: string;
  model: string;
  usage?: {
    input_tokens: number;
    output_tokens: number;
  };
}

/**
 * Extract text content from Anthropic-compatible response.
 * The response content array may contain both "thinking" and "text" blocks;
 * we only extract the "text" blocks.
 */
function extractTextFromContent(content: Array<{ type: string; text?: string }>): string {
  const textBlocks = content.filter(block => block.type === 'text' && block.text);
  return textBlocks.map(block => block.text!).join('\n');
}

/**
 * Call MiniMax's Anthropic-compatible messages API.
 * API key is always passed via options.apiKey.
 */
export async function minimaxChatCompletion(options: MinimaxChatOptions): Promise<MinimaxChatResponse> {
  const model = options.model || DEFAULT_MODEL;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120_000); // 2 min timeout

  // Separate system message from other messages for the Anthropic format
  const systemMessage = options.messages.find(m => m.role === 'system')?.content;
  const nonSystemMessages = options.messages.filter(m => m.role !== 'system');

  try {
    const body: Record<string, unknown> = {
      model,
      messages: nonSystemMessages,
      max_tokens: options.maxTokens ?? 2048,
      temperature: options.temperature ?? 0.7,
    };
    if (systemMessage) {
      body.system = systemMessage;
    }

    const response = await fetch(`${MINIMAX_BASE_URL}/v1/messages`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': options.apiKey,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    clearTimeout(timeout);

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`MiniMax API error (${response.status}): ${errText}`);
    }

    const data = await response.json();
    const content = extractTextFromContent(data.content || []);
    if (!content) {
      throw new Error('MiniMax API returned empty response');
    }

    return {
      content,
      model: data.model || model,
      usage: data.usage,
    };
  } catch (err: unknown) {
    clearTimeout(timeout);
    if (err instanceof Error && err.name === 'AbortError') {
      throw new Error('MiniMax API request timed out after 120s');
    }
    throw err;
  }
}

/**
 * Convenience: Call with system prompt + user content + API key.
 * This is the primary way the pipeline calls the LLM.
 */
export async function callMinimaxLLM(
  systemPrompt: string,
  userContent: string,
  apiKey: string,
  model?: string,
  maxTokens: number = 2048,
  temperature: number = 0.3
): Promise<string> {
  const modelName = model || DEFAULT_MODEL;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120_000);

  try {
    const response = await fetch(`${MINIMAX_BASE_URL}/v1/messages`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify({
        model: modelName,
        system: systemPrompt,
        messages: [
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
      throw new Error(`MiniMax API error (${response.status}): ${errText}`);
    }

    const data = await response.json();
    const content = extractTextFromContent(data.content || []);
    if (!content) {
      console.error('[MiniMax] Empty response. Full data:', JSON.stringify(data).substring(0, 500));
      throw new Error('MiniMax API returned empty response');
    }
    console.log(`[MiniMax] Response received. Content length: ${content.length}, preview: "${content.substring(0, 150)}"`);
    return content;
  } catch (err: unknown) {
    clearTimeout(timeout);
    if (err instanceof Error && err.name === 'AbortError') {
      throw new Error('MiniMax API request timed out after 120s');
    }
    throw err;
  }
}

export { DEFAULT_MODEL, MINIMAX_BASE_URL };
