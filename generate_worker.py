import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / 'data' / 'financial_chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

with open(BASE_DIR / 'baselines' / 'baseline.json', 'r', encoding='utf-8') as f:
    baseline = json.load(f)

with open(BASE_DIR / 'data' / 'online_eval_results.json', 'r', encoding='utf-8') as f:
    online_logs = json.load(f)

with open(BASE_DIR / 'goldens' / 'curated_flywheel_cases.json', 'r', encoding='utf-8') as f:
    flywheel = json.load(f)

worker_code = """
const FINANCIAL_CHUNKS = """ + json.dumps(chunks, indent=2) + """;
const BASELINE_DATA = """ + json.dumps(baseline, indent=2) + """;
const ONLINE_LOGS = """ + json.dumps(online_logs, indent=2) + """;
const FLYWHEEL_CASES = """ + json.dumps(flywheel, indent=2) + """;

const SYSTEM_PROMPT = `You are FinSecure AI, an enterprise SEC financial analyst assistant.
Your duty is to answer financial queries strictly and faithfully based on the official transcripts and disclosures provided in the context below.

STRICT OPERATIONAL GUIDELINES:
1. ONLY utilize verifiable facts, figures, and forward guidance directly quoted or stated within the <sec_filing_context>.
2. If the context does not contain sufficient information to answer the question with certainty, state clearly: "I cannot provide an answer because this information is not disclosed in the provided official filing documents."
3. Under NO circumstances disclose Material Non-Public Information (MNPI), personal insider information, rumors, or speculate beyond disclosed numbers.
4. Maintain a strictly professional, neutral, objective financial analyst tone.
5. Cite the company name and any specific revenue or margin percentages where applicable.`;

function bm25Retrieve(query, topK = 3) {
  const queryTokens = query.toLowerCase().split(/\\s+/).filter(w => w.length > 2);
  const scored = FINANCIAL_CHUNKS.map(chunk => {
    const textLower = chunk.text.toLowerCase();
    let score = 0;
    queryTokens.forEach(token => {
      const occurrences = (textLower.match(new RegExp(token, 'g')) || []).length;
      if (occurrences > 0) score += 1 + Math.log(1 + occurrences);
    });
    return { ...chunk, score };
  });
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK);
}

// Asynchronously stream nested traces to LangSmith
async function logToLangSmith(env, traceData) {
  const apiKey = env.LANGSMITH_API_KEY;
  if (!apiKey) return;

  const runId = crypto.randomUUID();
  const startTime = new Date(traceData.startTime).toISOString();
  const endTime = new Date(traceData.endTime).toISOString();
  const projectName = env.LANGSMITH_PROJECT || "RAG-EVAL-PROJECT";

  try {
    // 1. Post parent chain run
    const postPayload = {
      id: runId,
      name: "finsecure_rag_production_query",
      run_type: "chain",
      session_name: projectName,
      inputs: { query: traceData.query },
      outputs: { 
        answer: traceData.answer,
        retrieved_chunks_count: traceData.retrieved.length
      },
      start_time: startTime,
      end_time: endTime,
      extra: {
        metadata: {
          client: "cloudflare_pages_web_app",
          ai_gateway: env.AI_GATEWAY_ID || "finsecure-gateway",
          provider_used: traceData.providerUsed,
          retrieval_latency_sec: traceData.retrievalLatency,
          generation_latency_sec: traceData.generationLatency,
          total_latency_sec: traceData.totalLatency
        }
      }
    };

    const resp = await fetch("https://api.smith.langchain.com/runs", {
      method: "POST",
      headers: {
        "x-api-key": apiKey,
        "Content-Type": "application/json"
      },
      body: JSON.stringify(postPayload)
    });

    console.log("LangSmith trace published:", resp.status);
  } catch (err) {
    console.error("LangSmith background logging error:", err);
  }
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    // CORS Headers
    const corsHeaders = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization",
    };

    if (request.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders });
    }

    // Route: Telemetry / Baseline stats
    if (url.pathname === "/api/telemetry") {
      return new Response(JSON.stringify({
        baseline: BASELINE_DATA,
        online_logs: ONLINE_LOGS,
        flywheel: FLYWHEEL_CASES,
        gateway: {
          account_id: env.CLOUDFLARE_ACCOUNT_ID,
          gateway_id: env.AI_GATEWAY_ID,
          status: "ACTIVE"
        }
      }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" }
      });
    }

    // Route: RAG Query via Cloudflare AI Gateway + Groq LPU
    if (url.pathname === "/api/query" && request.method === "POST") {
      const startMs = Date.now();
      const body = await request.json().catch(() => ({}));
      const query = body.query || "What was NVIDIA revenue?";
      const topK = body.top_k || 3;

      // 1. Hybrid / Lexical Retrieval inside Worker
      const retrievalStartMs = Date.now();
      const retrieved = bm25Retrieve(query, topK);
      const retrievalLatency = (Date.now() - retrievalStartMs) / 1000;

      const contextText = retrieved.map(r => `[Source: ${r.company} (${r.ticker})]\\n${r.text}`).join("\\n\\n---\\n\\n");

      // 2. Multi-Provider Hybrid Inference (Groq LPU -> NVIDIA NIM -> OmniRoute)
      const accountId = env.CLOUDFLARE_ACCOUNT_ID || "4d39902ff31c1ac0bb2ef2ec2c637864";
      const gatewayId = env.AI_GATEWAY_ID || "finsecure-gateway";
      const groqKey = env.GROQ_API_KEY;
      const nvidiaKey = env.NVIDIA_API_KEY;
      const omnirouteKey = env.OMNIROUTE_API_KEY;
      const omnirouteBaseUrl = env.OMNIROUTE_BASE_URL || "https://api.omniroute.online/v1";

      const gatewayUrl = `https://gateway.ai.cloudflare.com/v1/${accountId}/${gatewayId}/groq/openai/v1/chat/completions`;

      const genStartMs = Date.now();
      let answer = "";
      let providerUsed = "None";
      let gatewayStatus = "OK";

      // Step 2a: Try Primary (Groq via Cloudflare AI Gateway)
      if (groqKey) {
        try {
          const aiResp = await fetch(gatewayUrl, {
            method: "POST",
            headers: {
              "Authorization": `Bearer ${groqKey}`,
              "Content-Type": "application/json",
              "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
            body: JSON.stringify({
              model: "qwen/qwen3.8-27b",
              temperature: 0.0,
              messages: [
                { role: "system", content: SYSTEM_PROMPT },
                { role: "user", content: `<sec_filing_context>\\n${contextText}\\n</sec_filing_context>\\n\\n<analyst_question>\\n${query}\\n</analyst_question>\\n\\nProvide a rigorous, concise, and faithful financial assessment:` }
              ],
              max_tokens: 500
            })
          });

          if (aiResp.ok) {
            const aiData = await aiResp.json();
            answer = aiData.choices[0]?.message?.content || "";
            providerUsed = "Groq LPU (qwen/qwen3.8-27b) via Cloudflare AI Gateway";
          } else {
            console.warn("Groq Gateway returned error:", aiResp.status);
          }
        } catch (err) {
          console.warn("Groq Gateway fetch error:", err.message);
        }
      }

      // Step 2b: Fallback 1 to Secondary (NVIDIA NIM GPU)
      if (!answer && nvidiaKey) {
        try {
          console.log("Triggering fallback to NVIDIA NIM...");
          const nvResp = await fetch("https://integrate.api.nvidia.com/v1/chat/completions", {
            method: "POST",
            headers: {
              "Authorization": `Bearer ${nvidiaKey}`,
              "Content-Type": "application/json"
            },
            body: JSON.stringify({
              model: "meta/llama-3.2-11b-vision-instruct",
              temperature: 0.0,
              messages: [
                { role: "system", content: SYSTEM_PROMPT },
                { role: "user", content: `<sec_filing_context>\\n${contextText}\\n</sec_filing_context>\\n\\n<analyst_question>\\n${query}\\n</analyst_question>\\n\\nProvide a rigorous, concise, and faithful financial assessment:` }
              ],
              max_tokens: 500
            })
          });

          if (nvResp.ok) {
            const nvData = await nvResp.json();
            answer = nvData.choices[0]?.message?.content || "";
            providerUsed = "NVIDIA NIM (meta/llama-3.2-11b-vision-instruct)";
            gatewayStatus = "FALLBACK_NVIDIA";
          } else {
            console.warn("NVIDIA NIM returned error:", nvResp.status);
          }
        } catch (err) {
          console.warn("NVIDIA NIM fetch error:", err.message);
        }
      }

      // Step 2c: Fallback 2 to Tertiary (OmniRoute Gateway)
      if (!answer && omnirouteKey) {
        try {
          console.log("Triggering fallback to OmniRoute Gateway...");
          const omniResp = await fetch(`${omnirouteBaseUrl}/chat/completions`, {
            method: "POST",
            headers: {
              "Authorization": `Bearer ${omnirouteKey}`,
              "Content-Type": "application/json"
            },
            body: JSON.stringify({
              model: "openai/gpt-oss-120b",
              temperature: 0.0,
              messages: [
                { role: "system", content: SYSTEM_PROMPT },
                { role: "user", content: `<sec_filing_context>\\n${contextText}\\n</sec_filing_context>\\n\\n<analyst_question>\\n${query}\\n</analyst_question>\\n\\nProvide a rigorous, concise, and faithful financial assessment:` }
              ],
              max_tokens: 500
            })
          });

          if (omniResp.ok) {
            const omniData = await omniResp.json();
            answer = omniData.choices[0]?.message?.content || "";
            providerUsed = "OmniRoute Gateway (openai/gpt-oss-120b)";
            gatewayStatus = "FALLBACK_OMNIROUTE";
          }
        } catch (err) {
          console.warn("OmniRoute fetch error:", err.message);
        }
      }

      if (!answer) {
        answer = "Error: All multi-provider LLM gateways (Groq, NVIDIA NIM, OmniRoute) were unreachable or exhausted limits.";
        gatewayStatus = "ERROR";
      }

      const endMs = Date.now();
      const genLatency = (endMs - genStartMs) / 1000;
      const totalLatency = (endMs - startMs) / 1000;

      // Log trace to LangSmith asynchronously via ctx.waitUntil (Zero user latency)
      if (ctx && ctx.waitUntil) {
        ctx.waitUntil(logToLangSmith(env, {
          query,
          answer,
          retrieved,
          startTime: startMs,
          endTime: endMs,
          providerUsed,
          retrievalLatency,
          generationLatency: genLatency,
          totalLatency
        }));
      } else {
        logToLangSmith(env, {
          query,
          answer,
          retrieved,
          startTime: startMs,
          endTime: endMs,
          providerUsed,
          retrievalLatency,
          generationLatency: genLatency,
          totalLatency
        });
      }

      return new Response(JSON.stringify({
        query,
        answer,
        retrieved_docs: retrieved,
        telemetry: {
          total_latency_sec: totalLatency,
          retrieval_latency_sec: retrievalLatency,
          generation_latency_sec: genLatency,
          provider_used: providerUsed,
          ai_gateway_endpoint: gatewayUrl,
          gateway_status: gatewayStatus,
          langsmith_traced: true
        }
      }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" }
      });
    }

    // Default 404
    return new Response("FinSecure RAG Cloudflare Worker API. Endpoints: /api/query, /api/telemetry", {
      headers: corsHeaders
    });
  }
};
"""

with open(BASE_DIR / 'cloudflare_worker' / 'worker.js', 'w', encoding='utf-8') as f:
    f.write(worker_code)

print("Generated cloudflare_worker/worker.js with direct LangSmith tracing successfully!")
