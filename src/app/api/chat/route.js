/**
 * RAG Resume Chatbot — API Route
 *
 * Performs retrieval over pre-computed vectors using BM25-style keyword search,
 * then generates answers using Groq (production) or Ollama (local).
 */

import { streamText } from "ai";
import { createGroq } from "@ai-sdk/groq";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { readFileSync } from "fs";
import { join } from "path";

// ── Load vectors at module level (cached across warm invocations) ──
let vectorData = null;

function getVectorData() {
  if (vectorData) return vectorData;

  try {
    const filePath = join(process.cwd(), "data", "vectors.json");
    const raw = readFileSync(filePath, "utf-8");
    vectorData = JSON.parse(raw);
    console.log(
      `[RAG] Loaded ${vectorData.total_chunks} chunks from vectors.json`
    );
    return vectorData;
  } catch (err) {
    console.error("[RAG] Failed to load vectors.json:", err.message);
    return null;
  }
}

// ── BM25-style keyword retrieval ────────────────────────────────────
/**
 * Simple but effective keyword-based retrieval.
 * No ML model needed at query time — works on Vercel serverless.
 *
 * Scoring: TF (term frequency in chunk) * IDF (inverse document frequency)
 */
function retrieveChunks(query, topK = 5) {
  const data = getVectorData();
  if (!data || !data.chunks || data.chunks.length === 0) {
    return [];
  }

  // Tokenize query
  const queryTokens = tokenize(query);
  if (queryTokens.length === 0) return data.chunks.slice(0, topK);

  // Compute IDF for each query token
  const totalDocs = data.chunks.length;
  const idf = {};
  for (const token of queryTokens) {
    const docsWithToken = data.chunks.filter((chunk) =>
      tokenize(chunk.text).includes(token)
    ).length;
    // IDF with smoothing
    idf[token] = Math.log((totalDocs + 1) / (docsWithToken + 1)) + 1;
  }

  // Score each chunk
  const scored = data.chunks.map((chunk) => {
    const chunkTokens = tokenize(chunk.text);
    const chunkLength = chunkTokens.length;

    let score = 0;
    for (const token of queryTokens) {
      // Term frequency
      const tf = chunkTokens.filter((t) => t === token).length;
      // Normalized TF * IDF
      const normalizedTf = tf / (chunkLength || 1);
      score += normalizedTf * (idf[token] || 1);
    }

    // Bonus: exact phrase match
    const lowerText = chunk.text.toLowerCase();
    const lowerQuery = query.toLowerCase();
    if (lowerText.includes(lowerQuery)) {
      score += 5;
    }

    // Bonus: consecutive word matches
    const queryWords = lowerQuery.split(/\s+/);
    for (let i = 0; i < queryWords.length - 1; i++) {
      const bigram = queryWords[i] + " " + queryWords[i + 1];
      if (lowerText.includes(bigram)) {
        score += 1;
      }
    }

    return { ...chunk, score };
  });

  // Sort by score descending and take top K
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK);
}

function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 2 && !STOP_WORDS.has(t));
}

const STOP_WORDS = new Set([
  "the", "and", "for", "are", "but", "not", "you", "all",
  "can", "has", "her", "was", "one", "our", "out", "his",
  "its", "had", "how", "who", "did", "get", "let", "say",
  "she", "too", "use", "him", "may", "any", "new", "now",
  "old", "see", "way", "day", "did", "from", "have", "been",
  "more", "when", "will", "this", "that", "with", "what",
  "your", "some", "them", "than", "then", "they", "into",
  "also", "just", "about", "would", "there", "their", "which",
  "could", "other", "after", "those", "these", "being",
]);

// ── LLM Provider Setup ─────────────────────────────────────────
function getLLM() {
  const provider = process.env.LLM_PROVIDER || "groq";

  if (provider === "ollama") {
    const ollama = createOpenAICompatible({
      name: "ollama",
      baseURL: process.env.OLLAMA_BASE_URL || "http://localhost:11434/v1",
    });
    return ollama(process.env.OLLAMA_MODEL || "llama3.2");
  }

  // Default: Groq
  const groq = createGroq({
    apiKey: process.env.GROQ_API_KEY,
  });
  return groq(process.env.GROQ_MODEL || "qwen/qwen3.6-27b");
}

// ── API Route Handler ───────────────────────────────────────────
export async function POST(req) {
  try {
    const { messages } = await req.json();

    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return Response.json(
        { error: "Messages array is required" },
        { status: 400 }
      );
    }

    // Get the latest user message for retrieval
    const lastUserMessage = [...messages]
      .reverse()
      .find((m) => m.role === "user");

    if (!lastUserMessage) {
      return Response.json(
        { error: "No user message found" },
        { status: 400 }
      );
    }

    // Retrieve relevant resume chunks
    const relevantChunks = retrieveChunks(lastUserMessage.content, 5);
    const context = relevantChunks
      .map((chunk, i) => `[Chunk ${i + 1}]: ${chunk.text}`)
      .join("\n\n");

    // Build system prompt with RAG context
    const systemPrompt = `You are a helpful AI assistant that answers questions about someone's resume and professional background. You are friendly, concise, and accurate.

IMPORTANT RULES:
- ONLY answer based on the resume information provided in the CONTEXT below.
- If the question cannot be answered from the context, say so honestly — do NOT make up information.
- Format your responses nicely using markdown when appropriate (bullet points, bold for emphasis).
- Be conversational but professional.
- If asked about something not in the resume, politely say the information is not available in the resume.

RESUME CONTEXT:
${context || "No relevant resume data found. Let the user know you couldn't find relevant information in the resume."}`;

    // Stream response from LLM
    const model = getLLM();
    console.log("[RAG] Sending to LLM with", relevantChunks.length, "context chunks");

    const result = await streamText({
      model,
      system: systemPrompt,
      messages: messages.map((m) => ({
        role: m.role,
        content: m.content,
      })),
      providerOptions: {
        groq: {
          // Suppress thinking/reasoning for cleaner output
          reasoning_format: "hidden",
        },
      },
    });

    return result.toDataStreamResponse();
  } catch (err) {
    console.error("[RAG API Error]:", err.message);
    console.error("[RAG API Error Stack]:", err.stack);

    // Helpful error messages
    if (err.message?.includes("API key")) {
      return Response.json(
        {
          error:
            "Groq API key issue: " + err.message,
        },
        { status: 500 }
      );
    }

    if (err.message?.includes("ECONNREFUSED")) {
      return Response.json(
        {
          error:
            "Cannot connect to Ollama. Make sure Ollama is running on localhost:11434.",
        },
        { status: 500 }
      );
    }

    return Response.json(
      { error: err.message || "Internal server error. Check server logs for details." },
      { status: 500 }
    );
  }
}

