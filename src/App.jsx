import React, { useState, useRef, useCallback, useEffect } from "react";

/* ════════════════════════════════════════════════════════════════
   INSPECTEUR AO — RAG CCTP Analyzer
   Mistral AI Portfolio · Projet 5
   ════════════════════════════════════════════════════════════════ */

const T = {
  bg: "#0a0a0f",
  surface: "#12121a",
  surfaceHover: "#1a1a26",
  border: "#1e1e2e",
  borderActive: "#f97316",
  accent: "#f97316",
  accentDim: "rgba(249,115,22,0.12)",
  accentGlow: "rgba(249,115,22,0.25)",
  ink: "#e2e2e8",
  inkDim: "#8b8b9e",
  inkDark: "#55556a",
  success: "#22c55e",
  successDim: "rgba(34,197,94,0.12)",
  white: "#ffffff",
  font: "'JetBrains Mono', 'Fira Code', 'SF Mono', monospace",
  fontSans: "'DM Sans', 'Segoe UI', system-ui, sans-serif",
};

const PDFJS_CDN = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174";

async function extractTextFromPDF(file) {
  const pdfjsLib = await loadPDFJS();
  const arrayBuffer = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  const pages = [];
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    const text = content.items.map((item) => item.str).join(" ");
    if (text.trim()) pages.push({ page: i, text: text.trim() });
  }
  return pages;
}

let pdfjsLoaded = null;
function loadPDFJS() {
  if (pdfjsLoaded) return pdfjsLoaded;
  pdfjsLoaded = new Promise((resolve) => {
    if (window.pdfjsLib) {
      window.pdfjsLib.GlobalWorkerOptions.workerSrc = PDFJS_CDN + "/pdf.worker.min.js";
      return resolve(window.pdfjsLib);
    }
    const script = document.createElement("script");
    script.src = PDFJS_CDN + "/pdf.min.js";
    script.onload = () => {
      window.pdfjsLib.GlobalWorkerOptions.workerSrc = PDFJS_CDN + "/pdf.worker.min.js";
      resolve(window.pdfjsLib);
    };
    document.head.appendChild(script);
  });
  return pdfjsLoaded;
}

function chunkPages(pages, maxChunkSize, overlap) {
  maxChunkSize = maxChunkSize || 1500;
  overlap = overlap || 200;
  var chunks = [];
  for (var p = 0; p < pages.length; p++) {
    var words = pages[p].text.split(/\s+/);
    var start = 0;
    while (start < words.length) {
      var slice = words.slice(start, start + maxChunkSize);
      chunks.push({
        text: slice.join(" "),
        page: pages[p].page,
        index: chunks.length,
      });
      start += maxChunkSize - overlap;
    }
  }
  return chunks;
}

function searchChunks(chunks, query, topK) {
  topK = topK || 6;
  var queryTerms = query
    .toLowerCase()
    .replace(/[^a-zà-ÿ0-9\s]/g, "")
    .split(/\s+/)
    .filter(function(t) { return t.length > 2; });

  var scored = chunks.map(function(chunk) {
    var text = chunk.text.toLowerCase();
    var score = 0;
    for (var i = 0; i < queryTerms.length; i++) {
      var regex = new RegExp(queryTerms[i], "gi");
      var matches = text.match(regex);
      if (matches) score += matches.length;
    }
    return { text: chunk.text, page: chunk.page, index: chunk.index, score: score };
  });

  return scored
    .filter(function(c) { return c.score > 0; })
    .sort(function(a, b) { return b.score - a.score; })
    .slice(0, topK);
}

async function askMistral(apiKey, context, question, docName) {
  var systemPrompt = "Tu es l'Inspecteur AO, un assistant expert en analyse de documents d'appels d'offres publics (CCTP, CCAG, CCAP, etc.).\n\nTu analyses le document \"" + docName + "\" pour répondre aux questions de l'utilisateur.\n\nRÈGLES :\n- Réponds UNIQUEMENT à partir du contexte fourni\n- Si l'information n'est pas dans le contexte, dis-le clairement\n- Cite les numéros de page quand c'est pertinent\n- Sois précis, structuré et professionnel\n- Réponds en français\n\nCONTEXTE EXTRAIT DU DOCUMENT :\n" + context;

  var response = await fetch("https://api.mistral.ai/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": "Bearer " + apiKey,
    },
    body: JSON.stringify({
      model: "mistral-large-latest",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: question },
      ],
      temperature: 0.2,
      max_tokens: 2048,
    }),
  });

  if (!response.ok) {
    var err = {};
    try { err = await response.json(); } catch(e) { /* ignore */ }
    throw new Error(err.message || "Erreur API " + response.status);
  }

  var data = await response.json();
  return (data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content) || "Pas de réponse.";
}

function fallbackAnswer(context, question) {
  var lines = context.split("\n").filter(function(l) { return l.trim(); });
  var qTerms = question.toLowerCase().split(/\s+/).filter(function(t) { return t.length > 2; });
  var relevant = lines.filter(function(l) {
    var lower = l.toLowerCase();
    return qTerms.some(function(t) { return lower.includes(t); });
  });

  if (relevant.length === 0) {
    return "⚠️ **Mode démo (sans clef API Mistral)**\n\nJ'ai trouvé " + lines.length + " passages pertinents dans le document, mais sans la clef API Mistral, je ne peux pas générer une réponse intelligente.\n\nVoici un extrait brut du contexte récupéré :\n\n> " + lines.slice(0, 3).join("\n> ") + "\n\n*Ajoutez votre clef API Mistral pour obtenir des réponses complètes et analysées.*";
  }

  var items = relevant.slice(0, 5).map(function(l, i) {
    return "**" + (i + 1) + ".** " + l.slice(0, 300) + (l.length > 300 ? "…" : "");
  });

  return "⚠️ **Mode démo (sans clef API Mistral)**\n\nPassages les plus pertinents trouvés :\n\n" + items.join("\n\n") + "\n\n*Ajoutez votre clef API Mistral pour obtenir une analyse complète par Mistral Large.*";
}

var SUGGESTED = [
  "Quel est l'objet principal de ce marché ?",
  "Quels sont les lots ou catégories définis ?",
  "Quels sont les montants estimatifs par lot ?",
  "Quelles sont les exigences de souveraineté ?",
  "Quelle est la durée du marché ?",
  "Quels sont les critères de sélection des candidats ?",
];

function formatMarkdown(text) {
  if (!text) return "";
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/`(.+?)`/g, '<code style="background:rgba(249,115,22,0.1);padding:2px 6px;border-radius:3px;font-size:0.85em">$1</code>')
    .replace(/^&gt; (.+)$/gm, '<div style="border-left:3px solid #f97316;padding-left:12px;margin:8px 0;color:#8b8b9e">$1</div>')
    .replace(/\n/g, "<br/>");
}

export default function App() {
  var _useState1 = useState(""); var apiKey = _useState1[0]; var setApiKey = _useState1[1];
  var _useState2 = useState(null); var file = _useState2[0]; var setFile = _useState2[1];
  var _useState3 = useState(""); var docName = _useState3[0]; var setDocName = _useState3[1];
  var _useState4 = useState([]); var chunks = _useState4[0]; var setChunks = _useState4[1];
  var _useState5 = useState(0); var pageCount = _useState5[0]; var setPageCount = _useState5[1];
  var _useState6 = useState(false); var loading = _useState6[0]; var setLoading = _useState6[1];
  var _useState7 = useState(""); var loadingMsg = _useState7[0]; var setLoadingMsg = _useState7[1];
  var _useState8 = useState(""); var question = _useState8[0]; var setQuestion = _useState8[1];
  var _useState9 = useState([]); var chat = _useState9[0]; var setChat = _useState9[1];
  var _useState10 = useState(false); var asking = _useState10[0]; var setAsking = _useState10[1];
  var chatEndRef = useRef(null);
  var inputRef = useRef(null);

  useEffect(function() {
    if (chatEndRef.current) chatEndRef.current.scrollIntoView({ behavior: "smooth" });
  }, [chat]);

  var handleFile = useCallback(async function(e) {
    var f = e.target.files && e.target.files[0];
    if (!f || f.type !== "application/pdf") return;
    setFile(f);
    setDocName(f.name);
    setLoading(true);
    setChat([]);
    setLoadingMsg("Extraction du texte...");

    try {
      var pages = await extractTextFromPDF(f);
      setPageCount(pages.length);
      setLoadingMsg("Découpage en chunks (" + pages.length + " pages)...");
      var c = chunkPages(pages);
      setChunks(c);
      setLoadingMsg("");
      setLoading(false);
      setChat([
        {
          role: "system",
          content: "✅ **" + f.name + "** chargé avec succès\n\n📄 " + pages.length + " pages extraites → " + c.length + " chunks indexés\n\nPosez une question sur le document ou choisissez une suggestion ci-dessous.",
        },
      ]);
    } catch (err) {
      setLoading(false);
      setLoadingMsg("");
      setChat([{ role: "system", content: "❌ Erreur lors du traitement : " + err.message }]);
    }
  }, []);

  var handleAsk = useCallback(
    async function(q) {
      var query = q || question;
      if (!query.trim() || chunks.length === 0) return;
      setQuestion("");
      setAsking(true);

      setChat(function(prev) { return prev.concat([{ role: "user", content: query }]); });

      var relevant = searchChunks(chunks, query);
      var context = relevant.map(function(c) { return "[Page " + c.page + "] " + c.text; }).join("\n\n");

      var uniquePages = [];
      relevant.forEach(function(r) { if (uniquePages.indexOf(r.page) === -1) uniquePages.push(r.page); });
      var sourcesMsg = relevant.length > 0
        ? "📎 " + relevant.length + " passages trouvés (pages " + uniquePages.join(", ") + ")"
        : "⚠️ Aucun passage pertinent trouvé";

      try {
        var answer;
        if (apiKey.trim()) {
          answer = await askMistral(apiKey, context, query, docName);
        } else {
          answer = fallbackAnswer(context, query);
        }
        setChat(function(prev) { return prev.concat([{ role: "assistant", content: answer, sources: sourcesMsg }]); });
      } catch (err) {
        setChat(function(prev) { return prev.concat([{ role: "assistant", content: "❌ Erreur : " + err.message, sources: sourcesMsg }]); });
      }

      setAsking(false);
      setTimeout(function() { if (inputRef.current) inputRef.current.focus(); }, 100);
    },
    [question, chunks, apiKey, docName]
  );

  return (
    <div style={styles.root}>
      <header style={styles.header}>
        <div style={styles.headerLeft}>
          <div style={styles.logoMark}>🔍</div>
          <div>
            <h1 style={styles.title}>INSPECTEUR AO</h1>
            <p style={styles.subtitle}>Analyse intelligente de documents d'appels d'offres</p>
          </div>
        </div>
        <div style={styles.headerRight}>
          <span style={styles.badge}>Mistral AI Portfolio</span>
          <span style={styles.badgeAlt}>RAG · Projet 5</span>
        </div>
      </header>

      <div style={styles.main}>
        <aside style={styles.sidebar}>
          <div style={styles.sideSection}>
            <label style={styles.sideLabel}>CLEF API MISTRAL</label>
            <input type="password" placeholder="sk-... (optionnel)" value={apiKey} onChange={function(e) { setApiKey(e.target.value); }} style={styles.sideInput} />
            <div style={styles.sideHint}>{apiKey ? "✅ Clef configurée" : "⚡ Mode démo sans clef"}</div>
          </div>

          <div style={styles.sideSection}>
            <label style={styles.sideLabel}>DOCUMENT PDF</label>
            <label style={styles.uploadBtn}>
              <input type="file" accept="application/pdf" onChange={handleFile} style={{ display: "none" }} />
              {file ? "📄 Changer de document" : "📂 Charger un CCTP / CCAG"}
            </label>
            {file && (
              <div style={styles.fileInfo}>
                <div style={styles.fileName}>{docName}</div>
                <div style={styles.fileStats}>{pageCount} pages · {chunks.length} chunks</div>
              </div>
            )}
          </div>

          <div style={styles.sideSection}>
            <label style={styles.sideLabel}>COMMENT ÇA MARCHE</label>
            {[
              "Upload d'un PDF (CCTP, CCAG…)",
              "Extraction & découpage en chunks",
              "Recherche sémantique des passages",
              "Mistral Large analyse & répond",
            ].map(function(txt, i) {
              return (
                <div key={i} style={styles.howStep}>
                  <span style={styles.stepNum}>{i + 1}</span>
                  <span style={styles.stepText}>{txt}</span>
                </div>
              );
            })}
          </div>

          <div style={styles.sideSection}>
            <label style={styles.sideLabel}>STACK TECHNIQUE</label>
            <div style={styles.techStack}>
              {["React", "PDF.js", "Mistral Large API", "TF-IDF Retrieval"].map(function(t) {
                return <span key={t} style={styles.techTag}>{t}</span>;
              })}
            </div>
          </div>
        </aside>

        <section style={styles.chatArea}>
          <div style={styles.chatMessages}>
            {chat.length === 0 && !loading && (
              <div style={styles.emptyState}>
                <div style={styles.emptyIcon}>📋</div>
                <div style={styles.emptyTitle}>Chargez un document pour commencer</div>
                <div style={styles.emptyDesc}>Uploadez un CCTP, CCAG, CCAP ou tout document d'appel d'offres au format PDF. L'Inspecteur AO l'analysera et répondra à vos questions grâce au RAG propulsé par Mistral.</div>
              </div>
            )}

            {loading && (
              <div style={styles.loadingBox}>
                <div style={styles.spinner}></div>
                <span>{loadingMsg}</span>
              </div>
            )}

            {chat.map(function(msg, i) {
              return (
                <div key={i} style={Object.assign({}, styles.msgRow, { justifyContent: msg.role === "user" ? "flex-end" : "flex-start" })}>
                  <div style={Object.assign({}, styles.msgBubble, msg.role === "user" ? styles.msgUser : msg.role === "system" ? styles.msgSystem : styles.msgAssistant)}>
                    {msg.role === "assistant" && <div style={styles.msgLabel}>🔍 Inspecteur AO</div>}
                    <div style={styles.msgContent} dangerouslySetInnerHTML={{ __html: formatMarkdown(msg.content) }}></div>
                    {msg.sources && <div style={styles.msgSources}>{msg.sources}</div>}
                  </div>
                </div>
              );
            })}

            {asking && (
              <div style={styles.msgRow}>
                <div style={Object.assign({}, styles.msgBubble, styles.msgAssistant)}>
                  <div style={styles.msgLabel}>🔍 Inspecteur AO</div>
                  <div style={styles.typingDots}>
                    <span style={styles.dot}></span>
                    <span style={Object.assign({}, styles.dot, { animationDelay: "0.2s" })}></span>
                    <span style={Object.assign({}, styles.dot, { animationDelay: "0.4s" })}></span>
                  </div>
                </div>
              </div>
            )}

            <div ref={chatEndRef}></div>
          </div>

          {chunks.length > 0 && chat.length <= 2 && (
            <div style={styles.suggestions}>
              {SUGGESTED.map(function(q) {
                return (
                  <button key={q} onClick={function() { handleAsk(q); }} style={styles.suggestionBtn} disabled={asking}>
                    {q}
                  </button>
                );
              })}
            </div>
          )}

          <div style={styles.inputBar}>
            <input
              ref={inputRef}
              type="text"
              placeholder={chunks.length === 0 ? "Chargez d'abord un document PDF…" : "Posez une question sur le document…"}
              value={question}
              onChange={function(e) { setQuestion(e.target.value); }}
              onKeyDown={function(e) { if (e.key === "Enter") handleAsk(); }}
              disabled={chunks.length === 0 || asking}
              style={styles.input}
            />
            <button
              onClick={function() { handleAsk(); }}
              disabled={chunks.length === 0 || asking || !question.trim()}
              style={Object.assign({}, styles.sendBtn, { opacity: (chunks.length === 0 || asking || !question.trim()) ? 0.4 : 1 })}
            >
              Analyser →
            </button>
          </div>
        </section>
      </div>

      <footer style={styles.footer}>
        <span>Inspecteur AO — Analyse RAG de documents d'appels d'offres</span>
        <span style={{ color: T.inkDark }}>Mistral AI Portfolio · Projet démonstratif</span>
      </footer>

      <style>{"\
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');\
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }\
        body { background: " + T.bg + "; color: " + T.ink + "; font-family: " + T.fontSans + "; -webkit-font-smoothing: antialiased; }\
        input:focus, button:focus { outline: none; }\
        ::placeholder { color: " + T.inkDark + "; }\
        ::-webkit-scrollbar { width: 6px; }\
        ::-webkit-scrollbar-track { background: transparent; }\
        ::-webkit-scrollbar-thumb { background: " + T.border + "; border-radius: 3px; }\
        @keyframes spin { to { transform: rotate(360deg); } }\
        @keyframes dotPulse { 0%, 80%, 100% { opacity: 0.3; transform: scale(0.8); } 40% { opacity: 1; transform: scale(1); } }\
        @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }\
      "}</style>
    </div>
  );
}

var styles = {
  root: { minHeight: "100vh", display: "flex", flexDirection: "column" },
  header: { display: "flex", alignItems: "center", justifyContent: "space-between", padding: "16px 28px", borderBottom: "1px solid " + T.border, background: T.surface },
  headerLeft: { display: "flex", alignItems: "center", gap: 14 },
  logoMark: { fontSize: 28, width: 48, height: 48, display: "flex", alignItems: "center", justifyContent: "center", background: T.accentDim, borderRadius: 12 },
  title: { fontSize: 18, fontWeight: 700, fontFamily: T.font, letterSpacing: "0.08em", color: T.white },
  subtitle: { fontSize: 12, color: T.inkDim, marginTop: 2 },
  headerRight: { display: "flex", gap: 8, alignItems: "center" },
  badge: { fontSize: 10, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", padding: "5px 10px", borderRadius: 6, background: T.accentDim, color: T.accent, fontFamily: T.font },
  badgeAlt: { fontSize: 10, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", padding: "5px 10px", borderRadius: 6, background: T.successDim, color: T.success, fontFamily: T.font },
  main: { flex: 1, display: "flex", overflow: "hidden" },
  sidebar: { width: 280, minWidth: 280, borderRight: "1px solid " + T.border, background: T.surface, padding: "20px 16px", display: "flex", flexDirection: "column", gap: 24, overflowY: "auto" },
  sideSection: { display: "flex", flexDirection: "column", gap: 8 },
  sideLabel: { fontSize: 10, fontWeight: 700, letterSpacing: "0.15em", textTransform: "uppercase", color: T.inkDark, fontFamily: T.font },
  sideInput: { background: T.bg, border: "1px solid " + T.border, borderRadius: 8, padding: "10px 12px", color: T.ink, fontSize: 13, fontFamily: T.font },
  sideHint: { fontSize: 11, color: T.inkDim },
  uploadBtn: { display: "block", textAlign: "center", padding: "12px 16px", borderRadius: 8, border: "2px dashed " + T.border, color: T.inkDim, fontSize: 13, fontWeight: 500, cursor: "pointer" },
  fileInfo: { padding: "10px 12px", background: T.accentDim, borderRadius: 8, border: "1px solid rgba(249,115,22,0.2)" },
  fileName: { fontSize: 12, fontWeight: 600, color: T.accent, wordBreak: "break-all" },
  fileStats: { fontSize: 11, color: T.inkDim, marginTop: 4 },
  howStep: { display: "flex", alignItems: "center", gap: 10 },
  stepNum: { width: 22, height: 22, borderRadius: "50%", background: T.accentDim, color: T.accent, fontSize: 11, fontWeight: 700, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, fontFamily: T.font },
  stepText: { fontSize: 12, color: T.inkDim, lineHeight: 1.4 },
  techStack: { display: "flex", flexWrap: "wrap", gap: 6 },
  techTag: { fontSize: 10, padding: "4px 8px", borderRadius: 4, background: T.bg, border: "1px solid " + T.border, color: T.inkDim, fontFamily: T.font },
  chatArea: { flex: 1, display: "flex", flexDirection: "column", background: T.bg },
  chatMessages: { flex: 1, overflowY: "auto", padding: "24px 28px", display: "flex", flexDirection: "column", gap: 16 },
  emptyState: { flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", textAlign: "center", gap: 12, padding: 40 },
  emptyIcon: { fontSize: 48, opacity: 0.4 },
  emptyTitle: { fontSize: 18, fontWeight: 600, color: T.inkDim },
  emptyDesc: { fontSize: 13, color: T.inkDark, maxWidth: 420, lineHeight: 1.6 },
  loadingBox: { display: "flex", alignItems: "center", gap: 12, padding: "14px 18px", background: T.surface, borderRadius: 10, color: T.inkDim, fontSize: 13, animation: "fadeIn 0.3s ease" },
  spinner: { width: 18, height: 18, border: "2px solid " + T.border, borderTopColor: T.accent, borderRadius: "50%", animation: "spin 0.8s linear infinite" },
  msgRow: { display: "flex", animation: "fadeIn 0.3s ease" },
  msgBubble: { maxWidth: "80%", borderRadius: 12, padding: "14px 18px", fontSize: 14, lineHeight: 1.7 },
  msgUser: { background: T.accentDim, border: "1px solid rgba(249,115,22,0.2)", color: T.ink, borderBottomRightRadius: 4 },
  msgAssistant: { background: T.surface, border: "1px solid " + T.border, color: T.ink, borderBottomLeftRadius: 4 },
  msgSystem: { background: T.successDim, border: "1px solid rgba(34,197,94,0.2)", color: T.ink, maxWidth: "100%" },
  msgLabel: { fontSize: 11, fontWeight: 700, color: T.accent, marginBottom: 8, fontFamily: T.font, letterSpacing: "0.05em" },
  msgContent: {},
  msgSources: { marginTop: 10, paddingTop: 8, borderTop: "1px solid " + T.border, fontSize: 11, color: T.inkDark, fontFamily: T.font },
  typingDots: { display: "flex", gap: 4, padding: "4px 0" },
  dot: { width: 8, height: 8, borderRadius: "50%", background: T.accent, animation: "dotPulse 1.2s infinite ease-in-out" },
  suggestions: { padding: "0 28px 16px", display: "flex", flexWrap: "wrap", gap: 8 },
  suggestionBtn: { padding: "8px 14px", borderRadius: 20, border: "1px solid " + T.border, background: "transparent", color: T.inkDim, fontSize: 12, cursor: "pointer", fontFamily: T.fontSans },
  inputBar: { padding: "16px 28px", borderTop: "1px solid " + T.border, display: "flex", gap: 10, background: T.surface },
  input: { flex: 1, padding: "12px 16px", borderRadius: 10, border: "1px solid " + T.border, background: T.bg, color: T.ink, fontSize: 14, fontFamily: T.fontSans },
  sendBtn: { padding: "12px 24px", borderRadius: 10, border: "none", background: T.accent, color: T.white, fontSize: 14, fontWeight: 600, cursor: "pointer", fontFamily: T.fontSans, whiteSpace: "nowrap" },
  footer: { padding: "12px 28px", borderTop: "1px solid " + T.border, display: "flex", justifyContent: "space-between", fontSize: 11, color: T.inkDim, background: T.surface },
};
