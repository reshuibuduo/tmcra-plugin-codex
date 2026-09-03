import { createHash, randomUUID } from "node:crypto";
import { createServer } from "node:http";

function json(response, status, body, headers = {}) {
  response.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    ...headers,
  });
  response.end(`${JSON.stringify(body)}\n`);
}

async function readJson(request) {
  let text = "";
  for await (const chunk of request) text += chunk;
  if (!text.trim()) return {};
  return JSON.parse(text);
}

function apiError(code, message, requestId) {
  return {
    error: {
      code,
      message,
      request_id: requestId,
    },
  };
}

function digest(value) {
  return createHash("sha256").update(String(value)).digest("hex").slice(0, 20);
}

export class MockTmcraServer {
  constructor({ validTokens = ["tmcra-test-valid-token"], ingestDelayMs = 0 } = {}) {
    this.validTokens = new Set(validTokens);
    this.ingestDelayMs = ingestDelayMs;
    this.records = [];
    this.jobs = new Map();
    this.idempotency = new Map();
    this.requests = [];
    this.server = null;
    this.baseUrl = null;
  }

  authorize(request) {
    const authorization = String(request.headers.authorization || "");
    const token = authorization.startsWith("Bearer ") ? authorization.slice(7) : "";
    return this.validTokens.has(token);
  }

  revoke(token) {
    this.validTokens.delete(token);
  }

  grant(token) {
    this.validTokens.add(token);
  }

  recordsForScope(scope) {
    return this.records.filter((record) => record.scope === scope);
  }

  async start() {
    if (this.server) return this;
    this.server = createServer((request, response) => {
      void this.handle(request, response).catch((error) => {
        json(response, 500, apiError("mock_internal_error", error.message, randomUUID()));
      });
    });
    await new Promise((resolve, reject) => {
      this.server.once("error", reject);
      this.server.listen(0, "localhost", resolve);
    });
    const address = this.server.address();
    this.baseUrl = `http://localhost:${address.port}`;
    return this;
  }

  async stop() {
    if (!this.server) return;
    const server = this.server;
    this.server = null;
    await new Promise((resolve, reject) => {
      server.close((error) => (error ? reject(error) : resolve()));
    });
  }

  async handle(request, response) {
    const requestId = `mock-${randomUUID()}`;
    const url = new URL(request.url || "/", this.baseUrl || "http://localhost");
    const requestRecord = {
      method: request.method,
      pathname: url.pathname,
      authorized: this.authorize(request),
      idempotencyKey: request.headers["idempotency-key"] || null,
      clientPlatform: request.headers["x-tmcra-client-platform"] || null,
    };
    this.requests.push(requestRecord);

    if (!requestRecord.authorized) {
      json(response, 401, apiError("invalid_token", "The API token is invalid or revoked.", requestId), {
        "x-request-id": requestId,
      });
      return;
    }

    if (url.pathname === "/v1/session" && request.method === "GET") {
      json(response, 200, {
        ok: true,
        authenticated: true,
        service: {
          name: "tmcra-memory",
          version: "0.2.0-mock",
          capabilities: [
            "ingest",
            "memory_graph",
            "quota_reporting",
            "recall",
            "scope_catalog",
          ],
        },
        credential: {
          type: "scope_token",
          tenant_id: "mock-tenant",
          principal: "subject:mock-personal-space",
          subject: "mock-personal-space",
          permissions: ["memory:read", "memory:write"],
          scope_restrictions: {
            unrestricted: false,
            names: [],
            prefixes: ["codex-e2e-"],
          },
          expires_at: 4_102_444_800,
        },
      });
      return;
    }

    const scopeMatch = url.pathname.match(/^\/v1\/scopes\/([^/]+)\/(recall|ingest)$/u);
    if (scopeMatch && request.method === "POST") {
      const scope = decodeURIComponent(scopeMatch[1]);
      if (scopeMatch[2] === "recall") {
        const body = await readJson(request);
        requestRecord.query = String(body.query || "");
        requestRecord.evidenceMode = String(body.evidence_mode || "");
        requestRecord.recallProfile = String(body.recall_profile || "quality");
        requestRecord.responseProjection = String(body.response_projection || "full");
        const records = this.recordsForScope(scope);
        const messages = records
          .flatMap((record) => record.messages)
          .map((message) => ({
            actor: String(message.role || "unknown").trim().toLowerCase(),
            text: String(message.content || "").trim(),
          }))
          .filter((message) => Boolean(message.text));
        const section = (actor, authority, title) => {
          const selected = messages.filter((message) => message.actor === actor);
          if (!selected.length) return "";
          const evidence = selected.map((message) => {
            return `[actor=${message.actor} | authority=${authority}]\n${message.text}`;
          }).join("\n\n");
          return `[TMCRA actor section | actor=${actor} | authority=${authority}]\n${title}\n\n${evidence}`;
        };
        const other = messages.filter(
          (message) => !["user", "assistant"].includes(message.actor),
        );
        const sections = [
          "[TMCRA authority policy | precedence=current_user>historical_user>assistant | assistant_is_not_user=true]",
          section("user", "user_statement", "User requirements and facts"),
          section(
            "assistant",
            "assistant_source",
            "Codex work progress and results (not user statements)",
          ),
          other.length
            ? `[TMCRA actor section | actors=mixed_or_unknown | authority=non_user]\nOther or mixed provenance (never user-authoritative)\n\n${other.map((message) => `[actor=${message.actor} | authority=non_user]\n${message.text}`).join("\n\n")}`
            : "",
        ].filter(Boolean);
        const content = sections.join("\n\n");
        json(response, 200, {
          query_id: `query-${digest(`${scope}:${body.query || ""}`)}`,
          scope_name: scope,
          evidence_route: "mock",
          prompt_evidence: {
            format: "text/plain",
            trust_boundary: "untrusted_memory_evidence",
            content,
          },
          evidence: {
            schema_version: "tmcra.memory_recall.v3.0",
            question: String(body.query || ""),
            evidence_windows: records.flatMap((record) => record.messages.map((message, index) => ({
              historical_date: "2026-07-19",
              timestamp: message.timestamp,
              message_role: message.role,
              rank: index + 1,
              text: message.content,
              unit_type: "source_window",
              role: ["source"],
              db_path: "/opt/tmcra/private/mock.sqlite3",
              scope_id: "internal-scope-id",
              retrieval_metadata: { raw_score: 999 },
            }))),
          },
        });
        return;
      }

      const body = await readJson(request);
      if (this.ingestDelayMs > 0) {
        await new Promise((resolve) => setTimeout(resolve, this.ingestDelayMs));
      }
      const idempotencyKey = String(request.headers["idempotency-key"] || "");
      if (idempotencyKey && this.idempotency.has(idempotencyKey)) {
        const jobId = this.idempotency.get(idempotencyKey);
        json(response, 200, { job_id: jobId, status: "queued", duplicate: true });
        return;
      }
      const jobId = `job-${randomUUID()}`;
      const record = {
        scope,
        sessionId: String(body.session_id || ""),
        messages: Array.isArray(body.messages) ? structuredClone(body.messages) : [],
        metadata: body.metadata && typeof body.metadata === "object" ? structuredClone(body.metadata) : {},
        consistency: body.consistency || null,
        slowPolicy: body.slow_policy || null,
        idempotencyKey: idempotencyKey || null,
        jobId,
      };
      this.records.push(record);
      this.jobs.set(jobId, {
        id: jobId,
        job_id: jobId,
        status: "succeeded",
        accepted_at: new Date().toISOString(),
        completed_at: Math.floor(Date.now() / 1000),
        tenant_id: "internal-tenant",
        scope_name: scope,
        result: {
          writer: {
            completed: true,
            input_messages: record.messages.length,
            new_message_count: record.messages.length,
            replayed_message_count: 0,
            batches: 1,
            validation_warnings: 0,
            db_path: "/opt/tmcra/private/native-memory.sqlite3",
          },
          index: {
            report: {
              status: "complete",
              candidate_count: record.messages.length,
              row_count: 1,
              elapsed_sec: 0.25,
              rows: [{ db_path: "/opt/tmcra/private/index.sqlite3" }],
            },
          },
        },
      });
      if (idempotencyKey) this.idempotency.set(idempotencyKey, jobId);
      json(response, 202, { job_id: jobId, status: "queued", duplicate: false });
      return;
    }

    const jobMatch = url.pathname.match(/^\/v1\/jobs\/([^/]+)$/u);
    if (jobMatch && request.method === "GET") {
      const job = this.jobs.get(decodeURIComponent(jobMatch[1]));
      if (!job) {
        json(response, 404, apiError("job_not_found", "Job was not found.", requestId));
        return;
      }
      json(response, 200, job);
      return;
    }

    json(response, 404, apiError("route_not_found", "Mock route was not found.", requestId));
  }
}
