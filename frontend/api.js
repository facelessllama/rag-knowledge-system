const API = 'http://localhost:8000';

async function apiHealth() {
  const r = await fetch(API + '/health');
  return r.ok;
}

async function apiGetModels() {
  const r = await fetch(API + '/models');
  if (!r.ok) throw new Error('models fetch failed');
  return r.json(); // { models: [...], current: string }
}

async function apiGetDocuments() {
  const r = await fetch(API + '/documents');
  if (!r.ok) throw new Error('documents fetch failed');
  return r.json(); // { documents: [...] }
}

async function apiUploadFile(file, folder) {
  const fd = new FormData();
  fd.append('file', file);
  if (folder) fd.append('folder', folder);
  const r = await fetch(API + '/upload', { method: 'POST', body: fd });
  return { ok: r.ok, status: r.status, data: r.ok ? await r.json() : null };
}

async function apiUploadBatch(files, folder) {
  const fd = new FormData();
  files.forEach(function(f) { fd.append('files', f); });
  if (folder) fd.append('folder', folder);
  const r = await fetch(API + '/upload-batch', { method: 'POST', body: fd });
  return { ok: r.ok, data: r.ok ? await r.json() : null };
}

async function apiDeleteDocument(docId) {
  const r = await fetch(API + '/documents/' + docId, { method: 'DELETE' });
  return r.ok;
}

async function apiQuery(question, topK, rerank, model) {
  const r = await fetch(API + '/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: question, top_k: topK, rerank: rerank, model: model || undefined })
  });
  return { ok: r.ok, data: r.ok ? await r.json() : null };
}

function apiQueryStream(question, topK, model, chatHistory, folder, onToken, onSources, onDone) {
  fetch(API + '/query/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: question, top_k: topK, model: model || undefined, chat_history: chatHistory || [], folder: folder || undefined })
  }).then(function(r) {
    if (!r.ok) { onDone(new Error('stream failed')); return; }
    var reader = r.body.getReader();
    var decoder = new TextDecoder();
    var buf = '';
    function read() {
      reader.read().then(function(result) {
        if (result.done) { onDone(null); return; }
        buf += decoder.decode(result.value, { stream: true });
        var lines = buf.split('\n');
        buf = lines.pop();
        lines.forEach(function(line) {
          if (!line.startsWith('data: ')) return;
          try {
            var ev = JSON.parse(line.slice(6));
            if (ev.type === 'token') onToken(ev.content);
            else if (ev.type === 'sources') onSources(ev.sources, ev.debug);
            else if (ev.type === 'done') onDone(null);
          } catch(e) {}
        });
        read();
      }).catch(function(e) { onDone(e); });
    }
    read();
  }).catch(function(e) { onDone(e); });
}

async function apiUpdateFolder(docId, folder) {
  const r = await fetch(API + '/documents/' + docId + '/folder', {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ folder: folder })
  });
  return r.ok;
}

async function apiGetHighlights(docId, text, page) {
  const r = await fetch(API + '/pdf/' + docId + '/highlights?page=' + page + '&text=' + encodeURIComponent(text));
  if (!r.ok) return null;
  return r.json(); // { rects, page_width, page_height }
}

function getPdfUrl(docId) {
  return API + '/pdf/' + docId;
}
