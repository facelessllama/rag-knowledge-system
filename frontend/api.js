const API = window.location.origin;

function authHeaders(extra) {
  const key = localStorage.getItem('api_key') || '';
  const h = Object.assign({ }, extra || {});
  if (key) h['X-API-Key'] = key;
  return h;
}

async function apiHealth() {
  const r = await fetch(API + '/health/ready');
  return r.ok;
}

async function apiGetModels() {
  const r = await fetch(API + '/models', { headers: authHeaders() });
  if (!r.ok) throw new Error('models fetch failed');
  return r.json(); // { models: [...], current: string }
}

async function apiGetDocuments() {
  const r = await fetch(API + '/documents', { headers: authHeaders() });
  if (!r.ok) throw new Error('documents fetch failed');
  return r.json(); // { documents: [...] }
}

async function apiGetDocumentPage(docId, pageNum) {
  const r = await fetch(API + '/documents/' + docId + '/pages/' + (pageNum || 1), { headers: authHeaders() });
  if (!r.ok) throw new Error('page fetch failed');
  return r.json(); // { document_id, format, page, total_pages, text, has_ocr }
}

async function apiUploadFile(file, folder) {
  const fd = new FormData();
  fd.append('file', file);
  if (folder) fd.append('folder', folder);
  const r = await fetch(API + '/upload', { method: 'POST', headers: authHeaders(), body: fd });
  return { ok: r.ok, status: r.status, data: r.ok ? await r.json() : null };
}

async function apiUploadBatch(files, folder) {
  const fd = new FormData();
  files.forEach(function(f) { fd.append('files', f); });
  if (folder) fd.append('folder', folder);
  const r = await fetch(API + '/upload-batch', { method: 'POST', headers: authHeaders(), body: fd });
  return { ok: r.ok, data: r.ok ? await r.json() : null };
}

async function apiDeleteDocument(docId) {
  const r = await fetch(API + '/documents/' + docId, { method: 'DELETE', headers: authHeaders() });
  return r.ok;
}

async function apiQuery(question, topK, rerank, model, provider, chatHistory, folder, documentIds) {
  const r = await fetch(API + '/query', {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({
      question: question, top_k: topK, rerank: rerank, model: model || undefined,
      // "local" is what the backend assumes when omitted — sent explicitly
      // anyway so it's never ambiguous which provider a request asked for.
      provider: provider || 'local',
      chat_history: chatHistory || [],
      folder: folder || undefined,
      document_ids: (documentIds && documentIds.length) ? documentIds : undefined,
    })
  });
  if (r.ok) return { ok: true, data: await r.json() };
  // Non-2xx still has a JSON body (e.g. ProviderNotAvailable's 400 detail) —
  // surface it instead of collapsing every failure into a generic
  // "no connection" message the user can't act on.
  let error = null;
  try { error = (await r.json()).detail || null; } catch (e) {}
  return { ok: false, data: null, error: error };
}

// Returns an AbortController — callers that want to cancel a stream in
// flight (e.g. starting a new query, navigating away, tearing down the
// chat panel) call the returned handle's .abort() to actually stop the
// underlying fetch/reader instead of leaving it running unobserved.
function apiQueryStream(question, topK, model, chatHistory, folder, documentIds, onToken, onSources, onDone) {
  var controller = new AbortController();
  var finished = false;

  // Single completion gate: exactly one of onDone(null) / onDone(err) ever
  // reaches the caller, regardless of how many of the paths below (the SSE
  // 'done'/'error' event, the underlying stream's own result.done, a fetch
  // rejection, or an external abort) end up firing. Previously an SSE
  // 'done'/'error' event didn't stop the read loop — it kept calling
  // read(), so the stream's own natural close a moment later re-fired
  // onDone(null) a second time, potentially overwriting an error already
  // reported to the caller with a spurious "success".
  function finishOnce(error) {
    if (finished) return;
    finished = true;
    onDone(error);
  }

  fetch(API + '/query/stream', {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    signal: controller.signal,
    body: JSON.stringify({
      question: question, top_k: topK, model: model || undefined, chat_history: chatHistory || [],
      folder: folder || undefined,
      // Strict scope for "compare these specific documents" — see
      // rag/retriever.py::retrieve_expanded's document_ids param. Distinct
      // from `folder`, which only narrows the candidate pool, not restricts
      // to an exact set.
      document_ids: (documentIds && documentIds.length) ? documentIds : undefined,
    })
  }).then(function(r) {
    if (!r.ok) { finishOnce(new Error('stream failed')); return; }
    var reader = r.body.getReader();
    var decoder = new TextDecoder();
    var buf = '';
    function read() {
      if (finished) return; // a terminal event already fired while this call was queued
      reader.read().then(function(result) {
        if (finished) return; // fired while this read() was in flight (e.g. an external abort)
        if (result.done) { finishOnce(null); return; }
        buf += decoder.decode(result.value, { stream: true });
        var lines = buf.split('\n');
        buf = lines.pop();
        for (var i = 0; i < lines.length; i++) {
          var line = lines[i];
          if (!line.startsWith('data: ')) continue;
          try {
            var ev = JSON.parse(line.slice(6));
            if (ev.type === 'token') { onToken(ev.content); continue; }
            if (ev.type === 'sources') { onSources(ev.sources, ev.debug); continue; }
            if (ev.type === 'done') { finishOnce(null); return; }
            if (ev.type === 'error') {
              finishOnce(Object.assign(new Error(ev.message || 'Server error'), { partial: !!ev.partial }));
              return;
            }
          } catch(e) {}
        }
        read();
      }).catch(function(e) {
        if (e && e.name === 'AbortError') { finishOnce(Object.assign(new Error('aborted'), { aborted: true })); return; }
        finishOnce(e);
      });
    }
    read();
  }).catch(function(e) {
    if (e && e.name === 'AbortError') { finishOnce(Object.assign(new Error('aborted'), { aborted: true })); return; }
    finishOnce(e);
  });

  return controller;
}

async function apiUpdateFolder(docId, folder) {
  const r = await fetch(API + '/documents/' + docId + '/folder', {
    method: 'PATCH',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ folder: folder })
  });
  return r.ok;
}

function getPdfUrl(docId) {
  return API + '/pdf/' + docId;
}

// PDF.js fetches via XHR/fetch, so the key can go in a header like every
// other request instead of the URL — a ?key= query param lands in browser
// history and server access logs, which a header doesn't.
function getPdfLoadOptions(docId) {
  return { url: getPdfUrl(docId), httpHeaders: authHeaders() };
}
