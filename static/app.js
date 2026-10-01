/* Phase 6 browser client for POST /ask.
 *
 * One deliberate security decision runs through this file: every piece of
 * model- or document-derived text is inserted with `textContent`, never
 * `innerHTML`. The answer text is LLM output and the chunk metadata comes from
 * crawled pages, so both are untrusted; assigning them as HTML would make this
 * page an XSS sink. The only nodes built from strings are elements, and the one
 * attribute set from data is a link `href`, which `safeUrl` restricts to
 * http(s) so a `javascript:` URL in a document field cannot execute.
 *
 * No conversation memory is claimed here. `/ask` is stateless - a follow-up is
 * answered as a standalone question - so this client keeps a transcript for the
 * reader and nothing more. (The Streamlit surface in app/ui.py does carry
 * `History`; that is a different process and a different guarantee.)
 */

const form = document.getElementById("ask-form");
const textarea = document.getElementById("question");
const sendBtn = document.getElementById("send");
const clearBtn = document.getElementById("clear");
const debugToggle = document.getElementById("debug-toggle");
const transcript = document.getElementById("transcript");
const intro = document.getElementById("intro");
const banner = document.getElementById("banner");

const DISCLAIMER = "Facts-only. No investment advice.";

/* A human label per guardrail/retrieval intent. `intent` is the value the
 * backend already computes, so the UI cannot invent a different story about why
 * a question was refused. */
const INTENT_LABELS = {
  answered: "Answered from source",
  no_context: "Not in my sources",
  unsupported: "Sources do not state this",
  advice: "No investment advice",
  pii: "Refused - personal data",
  injection: "Refused - unsafe request",
  off_topic: "Off topic",
  out_of_scope: "Out of scope",
  unsure: "Could not classify",
  degraded: "Temporarily unavailable",
  retrieval_error: "Search error",
  empty: "No question",
};

const ERROR_INTENTS = new Set(["degraded", "retrieval_error", "empty"]);

/** Only http(s) links are ever turned into an anchor. */
function safeUrl(value) {
  if (typeof value !== "string" || value === "") return null;
  try {
    const parsed = new URL(value, window.location.origin);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return null;
    return parsed.href;
  } catch {
    return null;
  }
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function newTurn(role) {
  const turn = el("div", `turn ${role}`);
  transcript.appendChild(turn);
  intro.hidden = true;
  return turn;
}

/** The user's own question, echoed locally. */
function renderQuestion(text) {
  const turn = newTurn("user");
  turn.appendChild(el("div", "bubble", text));
  transcript.scrollIntoView({ block: "end" });
  return turn;
}

/** A "Reading the source documents..." placeholder that is replaced in place. */
function renderPending() {
  const turn = newTurn("assistant");
  const bubble = el("div", "bubble");
  bubble.appendChild(el("span", "thinking", "Reading the source documents…"));
  turn.appendChild(bubble);
  turn.appendChild(el("p", "disclaimer-inline", DISCLAIMER));
  transcript.scrollIntoView({ block: "end" });
  return turn;
}

function renderError(text) {
  const turn = newTurn("assistant error");
  const bubble = el("div", "bubble", text);
  turn.appendChild(bubble);
  turn.appendChild(el("p", "disclaimer-inline", DISCLAIMER));
}

function addSource(body, citation) {
  const row = el("div", "source");
  const label = citation.title || citation.scheme || citation.host || "source";

  if (safeUrl(citation.url)) {
    const link = el("a", null, label);
    // The one place an attribute is written from retrieved data. safeUrl runs at
    // the assignment rather than into a variable first, so the value cannot be
    // swapped for an unvalidated one by a later edit.
    link.href = safeUrl(citation.url);
    link.target = "_blank";
    // noopener/noreferrer: the citation opens a third-party page, and we do not
    // want it reaching back through window.opener.
    link.rel = "noopener noreferrer";
    row.appendChild(link);
  } else {
    row.appendChild(el("strong", null, label));
  }

  const bits = [];
  if (citation.scheme) bits.push(citation.scheme);
  if (citation.page_type) bits.push(citation.page_type);
  if (citation.as_of) bits.push(`as of ${citation.as_of}`);
  if (bits.length) row.appendChild(el("div", "source-sub", bits.join(" · ")));

  // SC-3 is enforced server-side; this only surfaces a violation that slipped
  // through so it is visible rather than silently cited.
  if (citation.domain_ok === false) {
    row.appendChild(
      el("span", "warn-inline", "This link is not on the official-source allowlist.")
    );
  }
  body.appendChild(row);
}

function addRetrieval(body, retrieval) {
  const chunks = Array.isArray(retrieval.chunks) ? retrieval.chunks : [];
  if (!chunks.length) return;

  body.appendChild(
    el(
      "p",
      "note",
      `Read ${retrieval.n_chunks} of ${retrieval.n_candidates} candidate chunks ` +
        `(score threshold ${retrieval.min_score}).`
    )
  );

  const table = el("table", "hits");
  const head = el("tr");
  for (const label of ["rank", "score", "lexical", "page type", "as of"]) {
    head.appendChild(el("th", null, label));
  }
  table.appendChild(head);

  for (const chunk of chunks) {
    const row = el("tr");
    row.appendChild(el("td", "num", chunk.rank_score ?? ""));
    row.appendChild(el("td", "num", chunk.score ?? ""));
    row.appendChild(el("td", "num", chunk.lexical ?? ""));
    row.appendChild(el("td", null, chunk.page_type || ""));
    row.appendChild(el("td", null, chunk.as_of || ""));
    table.appendChild(row);
  }
  body.appendChild(table);
}

function renderAnswer(payload) {
  const turn = newTurn("assistant");
  const intent = payload.intent || "";
  const refused = payload.refused === true;

  turn.appendChild(el("div", "bubble", payload.answer || ""));

  const meta = el("div", "meta");
  const badgeClass = ERROR_INTENTS.has(intent) ? "error" : refused ? "refused" : "answered";
  meta.appendChild(el("span", `badge ${badgeClass}`, INTENT_LABELS[intent] || intent));
  if (typeof payload.elapsed_s === "number") {
    meta.appendChild(el("span", null, `${payload.elapsed_s.toFixed(2)}s`));
  }
  if (payload.repaired) meta.appendChild(el("span", null, "repaired to fit format"));
  turn.appendChild(meta);

  // Gate 6.1: repeated under every answer, not only in the header. A screenshot
  // of one answer is the artefact most likely to travel without its page.
  turn.appendChild(el("p", "disclaimer-inline", payload.disclaimer || DISCLAIMER));

  // Refusals carry a single educational link (`citation`); answers carry the
  // source document they were read from (`citations`). The two shapes are not
  // interchangeable: a refusal link is `{label, url}` - and `null` for a PII
  // refusal, which deliberately has nowhere to redirect the reader - while a
  // citation is a flat object with `title` and `url`. Reading `citation` as a
  // bare string is the bug this comment exists to prevent: safeUrl() rejects a
  // non-string, so the link silently renders as unclickable text.
  const links = [];
  if (Array.isArray(payload.citations)) {
    links.push(...payload.citations);
  } else if (payload.citation) {
    links.push(
      typeof payload.citation === "string"
        ? { url: payload.citation, title: "Educational reference" }
        : {
            url: payload.citation.url,
            title: payload.citation.label || "Educational reference",
          }
    );
  }

  if (links.length) {
    const details = el("details", "sources");
    details.appendChild(
      el("summary", null, links.length === 1 ? "Source (1)" : `Sources (${links.length})`)
    );
    const body = el("div", "sources-body");
    for (const citation of links) addSource(body, citation);
    if (payload.retrieval) addRetrieval(body, payload.retrieval);
    details.appendChild(body);
    turn.appendChild(details);
  } else if (payload.refused) {
    const details = el("details", "sources");
    details.appendChild(el("summary", null, "Sources (0)"));
    details.appendChild(
      el(
        "div",
        "sources-body",
        el(
          "p",
          "note",
          "No source. This is either a refusal - no document was read, so no " +
            "tokens were spent - or a case where the documents did not state the fact."
        )
      )
    );
    turn.appendChild(details);
  }

  transcript.scrollIntoView({ block: "end" });
}

let inFlight = false;

async function ask(question) {
  if (inFlight) return;
  inFlight = true;
  sendBtn.disabled = true;
  clearBtn.disabled = true;
  textarea.disabled = true;

  const pending = renderPending();

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question, debug: debugToggle.checked }),
    });

    let payload;
    try {
      payload = await response.json();
    } catch {
      throw new Error(`The server returned ${response.status} with a non-JSON body.`);
    }

    if (!response.ok) {
      // 4xx/5xx carries `detail`, and it is either a string or a list of
      // validation errors. Either way, show it rather than a generic failure.
      const detail = payload.detail ?? payload;
      const text =
        typeof detail === "string"
          ? detail
          : Array.isArray(detail)
            ? detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
            : "Something went wrong handling that question.";
      throw new Error(text);
    }

    pending.remove();
    renderAnswer(payload);
  } catch (err) {
    pending.remove();
    renderError(err && err.message ? err.message : "The request failed.");
  } finally {
    inFlight = false;
    sendBtn.disabled = false;
    clearBtn.disabled = false;
    textarea.disabled = false;
    textarea.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = textarea.value.trim();
  if (!question) return;
  textarea.value = "";
  renderQuestion(question);
  ask(question);
});

clearBtn.addEventListener("click", () => {
  transcript.replaceChildren();
  banner.hidden = true;
  intro.hidden = false;
  textarea.value = "";
  textarea.focus();
});

for (const chip of document.querySelectorAll(".chip")) {
  chip.addEventListener("click", () => {
    const question = chip.dataset.question;
    if (!question || inFlight) return;
    renderQuestion(question);
    ask(question);
  });
}

// Grow the composer with its content, up to the rows the textarea already has.
textarea.addEventListener("input", () => {
  textarea.style.height = "auto";
  textarea.style.height = `${Math.min(textarea.scrollHeight, 180)}px`;
});

// Surface a missing key or index once, on load, so a reviewer sees why the
// assistant cannot answer instead of discovering it one question at a time.
(async function reportStatus() {
  try {
    const response = await fetch("/healthz", { cache: "no-store" });
    if (!response.ok) return;
    const health = await response.json();
    const problems = [];
    if (!health.groq_configured) {
      problems.push("No Groq API key is configured, so no answer can be generated.");
    }
    if (!health.index_present) {
      problems.push("The vector index is missing, so no document can be read.");
    }
    if (problems.length) {
      banner.textContent = problems.join(" ");
      banner.hidden = false;
    }
  } catch {
    // A failed status check is not worth an error state: /ask still reports
    // every failure in its own response.
  }
})();
