let isTyping = false;
let docsData = {};
let currentModel = null;
let availableModels = [];
let activeFolderName = '';
const openFolders = new Set();
let chatHistory = [];
let currentLang = 'ru';

const LANGS = {
  ru: { flag: '🇷🇺', label: 'RU' },
  en: { flag: '🇬🇧', label: 'EN' },
};

function toggleLangDropdown() {
  var dd = document.getElementById('langDropdown');
  var btn = document.getElementById('langBtn');
  var show = !dd.classList.contains('show');
  dd.classList.toggle('show', show);
  btn.classList.toggle('open', show);
}

function selectLang(lang) {
  currentLang = lang;
  document.getElementById('langBtnFlag').textContent = LANGS[lang].flag;
  document.getElementById('langBtnLabel').textContent = LANGS[lang].label;
  document.getElementById('langOptRu').classList.toggle('active', lang === 'ru');
  document.getElementById('langOptEn').classList.toggle('active', lang === 'en');
  document.getElementById('langDropdown').classList.remove('show');
  document.getElementById('langBtn').classList.remove('open');
}

function esc(text) {
  const d = document.createElement('div');
  d.appendChild(document.createTextNode(String(text || '')));
  return d.innerHTML;
}
function scrollBottom() { const m = document.getElementById('messages'); m.scrollTop = m.scrollHeight; }
function hideEmpty() { const e = document.getElementById('emptyState'); if (e) e.remove(); }

// ── Health ────────────────────────────────────────────────────────────────────

async function checkHealth() {
  try {
    const ok = await apiHealth();
    document.getElementById('statusDot').style.background = ok ? '#1D9E75' : '#ef4444';
    document.getElementById('statusText').textContent = ok ? 'online' : 'error';
  } catch(e) {
    document.getElementById('statusDot').style.background = '#ef4444';
    document.getElementById('statusText').textContent = 'offline';
  }
}

// ── Models ────────────────────────────────────────────────────────────────────

async function loadModels() {
  try {
    const data = await apiGetModels();
    availableModels = data.models || [];
    currentModel = data.current;
    renderModelDropdown();
    updateModelBtn();
  } catch(e) {
    document.getElementById('modelBtnLabel').textContent = 'model';
  }
}

function renderModelDropdown() {
  const list = document.getElementById('modelList');
  if (!availableModels.length) {
    list.innerHTML = '<div class="model-loading">No models found</div>';
    return;
  }
  let html = '';
  availableModels.forEach(function(m) {
    html += '<div class="model-option ' + (m.name === currentModel ? 'active' : '') + '" onclick="selectModel(' + JSON.stringify(m.name).replace(/"/g, "'") + ')">';
    html += '<div class="model-option-name">' + esc(m.name) + '</div>';
    if (m.size_gb) html += '<div class="model-option-size">' + m.size_gb + 'GB</div>';
    html += '<span class="model-option-check">&#10003;</span></div>';
  });
  list.innerHTML = html;
}

function updateModelBtn() {
  const label = currentModel ? currentModel.replace(':latest', '') : 'model';
  document.getElementById('modelBtnLabel').textContent = label;
}

function selectModel(name) {
  currentModel = name;
  renderModelDropdown();
  updateModelBtn();
  closeModelDropdown();
}

function toggleModelDropdown() {
  const dd = document.getElementById('modelDropdown');
  const btn = document.getElementById('modelBtn');
  const show = !dd.classList.contains('show');
  dd.classList.toggle('show', show);
  btn.classList.toggle('open', show);
}

function closeModelDropdown() {
  document.getElementById('modelDropdown').classList.remove('show');
  document.getElementById('modelBtn').classList.remove('open');
}

document.addEventListener('click', function(e) {
  // Model dropdown close
  if (!document.getElementById('modelSelector').contains(e.target)) closeModelDropdown();

  // Lang dropdown close
  if (!document.getElementById('langSelector').contains(e.target)) {
    document.getElementById('langDropdown').classList.remove('show');
    document.getElementById('langBtn').classList.remove('open');
  }

  // Source card click
  var srcEl = e.target.closest('[data-src]');
  if (srcEl) {
    var s = _sourcesStore[parseInt(srcEl.dataset.src, 10)];
    if (s) openPdfViewer(s.document, s.page || 1, s.chunk_text || s.excerpt || '');
  }

  // Folder filter dropdown close / option select
  var dd = document.getElementById('folderFilterDropdown');
  var wrap = document.getElementById('folderFilterWrap');
  if (dd && wrap && !wrap.contains(e.target)) {
    dd.classList.remove('open');
    var btn = document.getElementById('folderFilterBtn');
    if (btn) btn.classList.toggle('active', !!_folderFilterValue);
  }
  var ffOpt = e.target.closest('[data-ff]');
  if (ffOpt && ffOpt.closest('#folderFilterDropdown')) {
    _folderFilterValue = ffOpt.dataset.ff;
    dd.classList.remove('open');
    updateFolderFilterSelect();
  }
});

// ── Folder tree ───────────────────────────────────────────────────────────────

function getFolderMap() {
  const map = {};
  Object.values(docsData).forEach(function(doc) {
    const f = doc.folder || 'Uncategorized';
    doc.folder = f;
    if (!map[f]) map[f] = [];
    if (!doc._placeholder) map[f].push(doc);
  });
  // Add empty folders (placeholders)
  Object.values(docsData).forEach(function(doc) {
    if (doc._placeholder) {
      const f = doc.folder;
      if (!map[f]) map[f] = [];
    }
  });
  return map;
}

function renderDocTree() {
  if (typeof updateFolderFilterSelect === 'function') updateFolderFilterSelect();
  const list = document.getElementById('docsList');
  const realDocs = Object.values(docsData).filter(function(d){ return !d._placeholder; });
  const total = realDocs.length;

  if (Object.keys(docsData).length === 0) {
    list.innerHTML = '<div class="empty-docs">No documents yet<br><span style="font-size:10px">Create a folder to start</span></div>';
    document.getElementById('compareBtn').style.display = 'none';
    document.getElementById('headerStats').innerHTML = '';
    return;
  }

  const map = getFolderMap();
  const names = Object.keys(map).sort(function(a, b) {
    if (a === 'Uncategorized') return 1;
    if (b === 'Uncategorized') return -1;
    return a.localeCompare(b);
  });

  let html = '';
  names.forEach(function(fname) {
    const docs = map[fname] || [];
    const isActive = fname === activeFolderName;
    const isOpen = openFolders.has(fname);
    const isUncategorized = fname === 'Uncategorized';
    const fnameJson = JSON.stringify(fname);

    html += '<div class="folder-group' + (isActive ? ' active' : '') + '">';

    html += '<div class="folder-header" onclick="toggleFolder(' + fnameJson.replace(/"/g, "'") + ')"'
      + ' ondragover="event.preventDefault();this.parentElement.style.outline=\'1px solid var(--accent)\'"'
      + ' ondragleave="this.parentElement.style.outline=\'\'"'
      + ' ondrop="dropOnFolder(event,' + fnameJson.replace(/"/g, "'") + ')">';
    html += '<span class="folder-arrow' + (isOpen ? ' open' : '') + '">&#9658;</span>';
    html += '<span class="folder-icon">' + (isOpen ? '&#128194;' : (isUncategorized ? '&#128203;' : '&#128193;')) + '</span>';
    html += '<span class="folder-name">' + esc(fname) + '</span>';
    html += '<span class="folder-count">' + docs.length + '</span>';
    html += '<div class="folder-actions">';
    html += '<button class="fld-btn upload-here" onclick="event.stopPropagation();uploadToFolder(' + fnameJson.replace(/"/g, "'") + ')" title="Upload PDFs here">&#8679;</button>';
    if (!isUncategorized) {
      html += '<button class="fld-btn" onclick="event.stopPropagation();renameFolder(' + fnameJson.replace(/"/g, "'") + ')" title="Rename">&#9998;</button>';
      html += '<button class="fld-btn del" id="del-btn-' + esc(fname) + '" onclick="event.stopPropagation();confirmDeleteFolder(event,' + fnameJson.replace(/"/g, "'") + ')" title="Delete">&#10005;</button>';
    }
    html += '</div></div>';

    if (isOpen) {
      html += '<div class="folder-files">';
      if (docs.length === 0) {
        html += '<div style="font-size:11px;color:var(--text-muted);padding:6px 8px;">Empty — upload PDFs here</div>';
      } else {
        docs.forEach(function(doc) { html += renderDocItem(doc); });
      }
      html += '</div>';
    }
    html += '</div>';
  });

  html += '<button class="new-folder-btn" onclick="createNewFolder()">&#65291; New folder</button>';
  list.innerHTML = html;

  list.querySelectorAll('.doc-delete').forEach(function(btn) {
    btn.addEventListener('click', function(e) {
      e.stopPropagation();
      const doc = docsData[btn.dataset.docId];
      if (doc) deleteDocument(btn.dataset.docId, doc.filename);
    });
  });
  list.querySelectorAll('.doc-item').forEach(function(item) {
    item.addEventListener('click', function() {
      const doc = docsData[item.dataset.docId];
      if (doc) openPdfViewer(item.dataset.docId, 1, '');
    });
  });

  document.getElementById('compareBtn').style.display = total >= 2 ? 'block' : 'none';
  const totalChunks = realDocs.reduce(function(s, d){ return s + (d.chunks || d.chunks_created || 0); }, 0);
  document.getElementById('headerStats').innerHTML = '<span>' + total + '</span> docs &middot; <span>' + totalChunks + '</span> chunks';
}

function renderDocItem(doc) {
  const chunks = doc.chunks_created || doc.chunks || 0;
  const pages = doc.pages || 0;
  return '<div class="doc-item" data-doc-id="' + doc.doc_id + '"'
    + ' draggable="true"'
    + ' ondragstart="event.dataTransfer.setData(\'docId\',\'' + doc.doc_id + '\');this.style.opacity=\'0.4\'"'
    + ' ondragend="this.style.opacity=\'1\'">'
    + '<span class="doc-icon">&#128203;</span>'
    + '<div class="doc-info">'
    + '<div class="doc-name" title="' + esc(doc.filename) + '">' + esc(doc.filename) + '</div>'
    + '<div class="doc-meta">' + pages + 'p &middot; ' + chunks + ' chunks</div>'
    + '</div>'
    + '<span class="doc-badge">ready</span>'
    + '<button class="doc-delete" data-doc-id="' + doc.doc_id + '" title="Delete">&#10005;</button>'
    + '</div>';
}

// ── Folder actions ────────────────────────────────────────────────────────────

function toggleFolder(fname) {
  if (openFolders.has(fname)) {
    openFolders.delete(fname);
  } else {
    openFolders.add(fname);
    setActiveFolder(fname);
  }
  renderDocTree();
}

function uploadToFolder(fname) {
  setActiveFolder(fname);
  document.getElementById('fileInput').click();
}

function setActiveFolder(fname) {
  activeFolderName = fname;
  const badge = document.getElementById('activeFolderBadge');
  const label = document.getElementById('activeFolderLabel');
  if (fname && fname !== 'Uncategorized') {
    label.textContent = fname;
    badge.classList.add('show');
  } else {
    badge.classList.remove('show');
  }
  renderDocTree();
}

function clearActiveFolder() {
  activeFolderName = '';
  document.getElementById('activeFolderBadge').classList.remove('show');
  renderDocTree();
}

function createNewFolder() {
  const name = prompt('Folder name:');
  if (!name || !name.trim()) return;
  const fname = name.trim();
  docsData['__ph__' + fname] = { doc_id: '__ph__' + fname, filename: '', folder: fname, _placeholder: true, pages: 0, chunks: 0 };
  openFolders.add(fname);
  setActiveFolder(fname);
  fetch(API + '/folders', { method: 'POST', headers: authHeaders({'Content-Type':'application/json'}), body: JSON.stringify({name: fname}) });
}

function renameFolder(oldName) {
  const newName = prompt('New name:', oldName);
  if (!newName || !newName.trim() || newName.trim() === oldName) return;
  const n = newName.trim();
  Object.values(docsData).forEach(function(doc) {
    if (doc.folder === oldName) {
      doc.folder = n;
      if (doc._placeholder) { doc.doc_id = '__ph__' + n; delete docsData['__ph__' + oldName]; docsData['__ph__' + n] = doc; }
    }
  });
  if (openFolders.has(oldName)) { openFolders.delete(oldName); openFolders.add(n); }
  if (activeFolderName === oldName) setActiveFolder(n); else renderDocTree();
  fetch(API + '/folders/' + encodeURIComponent(oldName), { method: 'PATCH', headers: authHeaders({'Content-Type':'application/json'}), body: JSON.stringify({name: n}) });
}

async function dropOnFolder(event, targetFolder) {
  event.preventDefault();
  event.currentTarget.parentElement.style.outline = '';
  const docId = event.dataTransfer.getData('docId');
  if (!docId || !docsData[docId]) return;
  docsData[docId].folder = targetFolder;
  renderDocTree();
  await apiUpdateFolder(docId, targetFolder);
}

let _deletePending = null;

function confirmDeleteFolder(event, fname) {
  event.stopPropagation();
  const btn = event.currentTarget;
  if (_deletePending === fname) {
    // Second click — confirmed
    _deletePending = null;
    deleteFolder(fname);
    return;
  }
  // First click — show confirmation state
  _deletePending = fname;
  btn.textContent = 'Delete?';
  btn.classList.add('del-confirm');
  // Auto-reset after 3s if no second click
  setTimeout(function() {
    if (_deletePending === fname) {
      _deletePending = null;
      renderDocTree();
    }
  }, 3000);
}

function deleteFolder(fname) {
  const docs = Object.values(docsData).filter(function(d){ return d.folder === fname && !d._placeholder; });
  docs.forEach(function(doc) { deleteDocument(doc.doc_id, doc.filename, true); });
  Object.keys(docsData).forEach(function(k) { if (docsData[k].folder === fname) delete docsData[k]; });
  openFolders.delete(fname);
  if (activeFolderName === fname) clearActiveFolder(); else renderDocTree();
  fetch('http://localhost:8000/folders/' + encodeURIComponent(fname), { method: 'DELETE' });
}

// ── Documents ─────────────────────────────────────────────────────────────────

var _folderFilterValue = '';

function updateFolderFilterSelect() {
  var folders = Object.keys(getFolderMap()).filter(function(f){ return f !== 'Uncategorized'; }).sort();
  var dd = document.getElementById('folderFilterDropdown');
  if (!dd) return;
  var html = '<div class="ff-dropdown-header">Search in</div>';
  html += '<div class="ff-option' + (!_folderFilterValue ? ' selected' : '') + '" data-ff=""><span>All folders</span><span class="ff-check">✓</span></div>';
  folders.forEach(function(f) {
    html += '<div class="ff-option' + (_folderFilterValue === f ? ' selected' : '') + '" data-ff="' + esc(f) + '"><span>' + esc(f) + '</span><span class="ff-check">✓</span></div>';
  });
  dd.innerHTML = html;
  // validate current selection
  if (_folderFilterValue && folders.indexOf(_folderFilterValue) === -1) {
    _folderFilterValue = '';
  }
  document.getElementById('folderFilterLabel').textContent = _folderFilterValue || 'All folders';
  var btn = document.getElementById('folderFilterBtn');
  if (btn) btn.classList.toggle('active', !!_folderFilterValue || dd.classList.contains('open'));
}

function toggleFolderFilter(e) {
  e.stopPropagation();
  var dd = document.getElementById('folderFilterDropdown');
  var btn = document.getElementById('folderFilterBtn');
  var opening = !dd.classList.contains('open');
  dd.classList.toggle('open', opening);
  // chevron rotates when open OR when a folder is selected
  btn.classList.toggle('active', opening || !!_folderFilterValue);
}


async function loadDocuments() {
  try {
    const data = await apiGetDocuments();
    docsData = {};
    (data.documents || []).forEach(function(doc) { docsData[doc.doc_id] = doc; });
    // Restore persisted empty folders as placeholders
    (data.folders || []).forEach(function(fname) {
      if (fname && !Object.values(docsData).some(function(d){ return d.folder === fname && !d._placeholder; })) {
        docsData['__ph__' + fname] = { doc_id: '__ph__' + fname, filename: '', folder: fname, _placeholder: true, pages: 0, chunks: 0 };
      }
    });
    updateFolderFilterSelect();
    renderDocTree();
  } catch(e) { console.log(e); }
}

// ── Progress helpers ──────────────────────────────────────────────────────────

function showProg() { document.getElementById('uploadProgress').style.display = 'block'; }
function hideProg(ms) {
  setTimeout(function() {
    document.getElementById('uploadProgress').style.display = 'none';
    document.getElementById('progressFill').style.width = '0%';
    document.getElementById('progressFill').style.background = 'var(--accent)';
  }, ms || 2000);
}
function setProg(pct, msg) {
  document.getElementById('progressFill').style.width = pct + '%';
  document.getElementById('progressText').textContent = msg;
}

// ── Upload ────────────────────────────────────────────────────────────────────

async function uploadFiles(input) {
  const files = Array.from(input.files).filter(function(f){ return f.name.toLowerCase().endsWith('.pdf'); });
  if (!files.length) return;
  const folder = (activeFolderName && activeFolderName !== 'Uncategorized') ? activeFolderName : '';
  showProg();

  if (files.length === 1) {
    setProg(30, 'Uploading...');
    try {
      setProg(60, 'Processing...');
      const res = await apiUploadFile(files[0], folder);
      setProg(100, '');
      if (res.ok) {
        const d = res.data;
        d.folder = folder || 'Uncategorized';
        docsData[d.doc_id] = d;
        if (folder) { const ph = '__ph__' + folder; if (docsData[ph]) delete docsData[ph]; }
        setProg(100, 'Done!');
        renderDocTree();
      } else if (res.status === 409) {
        setProg(100, 'Already uploaded');
        document.getElementById('progressFill').style.background = '#ca8a04';
      } else {
        setProg(100, 'Upload failed');
        document.getElementById('progressFill').style.background = '#ef4444';
      }
    } catch(e) { setProg(100, 'Error'); document.getElementById('progressFill').style.background = '#ef4444'; }
  } else {
    setProg(20, 'Uploading ' + files.length + ' files...');
    try {
      setProg(60, 'Processing...');
      const res = await apiUploadBatch(files, folder);
      if (res.ok) {
        const data = res.data;
        data.results.forEach(function(r) {
          if (r.status === 'indexed') { r.folder = folder || 'Uncategorized'; docsData[r.doc_id] = r; }
        });
        if (folder) { const ph = '__ph__' + folder; if (docsData[ph]) delete docsData[ph]; }
        setProg(100, 'Done: ' + data.indexed + ' indexed' + (data.skipped ? ', ' + data.skipped + ' skipped' : ''));
        renderDocTree();
      } else { setProg(100, 'Failed'); document.getElementById('progressFill').style.background = '#ef4444'; }
    } catch(e) { setProg(100, 'Error'); document.getElementById('progressFill').style.background = '#ef4444'; }
  }
  hideProg(2000);
  input.value = '';
}

async function uploadFolder(input) {
  const pdfFiles = Array.from(input.files).filter(function(f){ return f.name.toLowerCase().endsWith('.pdf'); });
  if (!pdfFiles.length) { alert('No PDF files found in folder'); input.value = ''; return; }

  showProg();
  const byFolder = {};
  pdfFiles.forEach(function(f) {
    const parts = f.webkitRelativePath.split('/');
    const fp = parts.length > 2 ? parts.slice(0, parts.length - 1).join(' / ') : parts[0];
    if (!byFolder[fp]) byFolder[fp] = [];
    byFolder[fp].push(f);
  });

  const folderList = Object.keys(byFolder);
  let done = 0, indexed = 0, skipped = 0;

  for (let i = 0; i < folderList.length; i++) {
    const fname = folderList[i];
    done++;
    setProg(Math.round(done / folderList.length * 90) + 5, fname + ' (' + byFolder[fname].length + ')...');
    try {
      const res = await apiUploadBatch(byFolder[fname], fname);
      if (res.ok) {
        const data = res.data;
        indexed += data.indexed || 0;
        skipped += data.skipped || 0;
        data.results.forEach(function(r) {
          if (r.status === 'indexed') { r.folder = fname; docsData[r.doc_id] = r; }
        });
        openFolders.add(fname);
      }
    } catch(e) { console.error(e); }
  }

  setProg(100, '&#10003; ' + indexed + ' indexed' + (skipped ? ', ' + skipped + ' skipped' : ''));
  renderDocTree();
  hideProg(3000);
  input.value = '';
}

async function deleteDocument(doc_id, filename, silent) {
  if (!silent && !confirm('Delete "' + filename + '"?')) return;
  try {
    const ok = await apiDeleteDocument(doc_id);
    if (ok) { delete docsData[doc_id]; if (!silent) renderDocTree(); }
  } catch(e) { if (!silent) alert('Connection error'); }
}

function compareDocuments() {
  const names = Object.values(docsData).filter(function(d){ return !d._placeholder && d.filename; }).map(function(d){ return d.filename; }).join(', ');
  document.getElementById('chatInput').value = 'What is the main topic of each document and how do they differ? Documents: ' + names;
  sendMessage();
}

function useSuggestion(el) { document.getElementById('chatInput').value = el.textContent; sendMessage(); }

// ── Messages ──────────────────────────────────────────────────────────────────

function addUserMessage(text) {
  hideEmpty();
  const c = document.getElementById('messages');
  const d = document.createElement('div');
  d.className = 'message user';
  d.innerHTML = '<div class="bubble">' + esc(text) + '</div>';
  c.appendChild(d); scrollBottom();
}

function showTyping() {
  hideEmpty();
  const c = document.getElementById('messages');
  const d = document.createElement('div');
  d.className = 'message bot'; d.id = 'typingIndicator';
  d.innerHTML = '<div class="typing-bubble"><span></span><span></span><span></span></div>';
  c.appendChild(d); scrollBottom();
}

function hideTyping() { const t = document.getElementById('typingIndicator'); if (t) t.remove(); }

function addBotMessage(text, sources) {
  const c = document.getElementById('messages');
  const d = document.createElement('div');
  d.className = 'message bot';
  let html = '<div class="bubble">' + esc(text) + '</div>';
  if (sources && sources.length > 0) {
    html += '<div class="sources"><div class="sources-label">Sources</div><div class="sources-grid">';
    sources.forEach(function(s) {
      const score = s.relevance_score ? Math.round(s.relevance_score * 100) + '%' : '';
      const doc = Object.values(docsData).find(function(x){ return x.doc_id === s.document; }) || {};
      const fname = doc.filename ? doc.filename.replace(/\.pdf$/i, '') : (s.document || '?');
      const short = fname.length > 22 ? fname.slice(0, 20) + '...' : fname;
      const docIdSafe = JSON.stringify(s.document || '').replace(/"/g, "'");
      const chunkSafe = JSON.stringify(s.chunk_text || s.excerpt || '').replace(/"/g, "'");
      const page = s.page || 1;
      html += '<div class="source-item" onclick="openPdfViewer(' + docIdSafe + ',' + page + ',' + chunkSafe + ')">';
      html += '<div class="source-meta">';
      html += '<span class="source-filename" title="' + esc(doc.filename || '') + '">' + esc(short) + '</span>';
      html += '<span class="source-page">p. ' + (s.page || '?') + '</span>';
      html += '</div>';
      html += '<span class="source-text">' + esc(s.excerpt || '') + '</span>';
      if (score) html += '<span class="source-score">' + score + '</span>';
      html += '<span style="font-size:10px;color:var(--accent);flex-shrink:0">&#8599;</span>';
      html += '</div>';
    });
    html += '</div></div>';
  }
  d.innerHTML = html; c.appendChild(d); scrollBottom();
}

function addErrorMessage(text) {
  const c = document.getElementById('messages');
  const d = document.createElement('div');
  d.className = 'message bot';
  d.innerHTML = '<div class="error-msg">' + esc(text) + '</div>';
  c.appendChild(d); scrollBottom();
}

// ── Sources store (avoids inline-onclick escaping bugs) ───────────────────────
var _sourcesStore = [];

function buildSourcesHtml(sources) {
  if (!sources || !sources.length) return '';
  const baseIdx = _sourcesStore.length;
  sources.forEach(function(s) { _sourcesStore.push(s); });

  let html = '<div class="sources"><div class="sources-label">Sources</div><div class="sources-grid">';
  sources.forEach(function(s, i) {
    const idx = baseIdx + i;
    const score = s.relevance_score ? Math.round(s.relevance_score * 100) + '%' : '';
    const doc = Object.values(docsData).find(function(x){ return x.doc_id === s.document; }) || {};
    const fname = doc.filename ? doc.filename.replace(/\.pdf$/i, '') : (s.document || '?');
    const short = fname.length > 22 ? fname.slice(0, 20) + '...' : fname;
    html += '<div class="source-item" data-src="' + idx + '">';
    html += '<div class="source-meta">';
    html += '<span class="source-filename" title="' + esc(doc.filename || '') + '">' + esc(short) + '</span>';
    html += '<span class="source-page">p. ' + (s.page || '?') + '</span>';
    html += '</div>';
    html += '<span class="source-text">' + esc(s.excerpt || '') + '</span>';
    if (score) html += '<span class="source-score">' + score + '</span>';
    html += '<span style="font-size:10px;color:var(--accent);flex-shrink:0">&#8599;</span>';
    html += '</div>';
  });
  html += '</div></div>';
  return html;
}

// ── Send ──────────────────────────────────────────────────────────────────────

function sendMessage() {
  if (isTyping) return;
  const input = document.getElementById('chatInput');
  const text = input.value.trim();
  if (!text) return;
  input.value = ''; isTyping = true;
  addUserMessage(text); showTyping();

  var accum = '';
  var msgEl = null;
  var bubbleEl = null;
  var tokenQueue = [];
  var draining = false;
  var pendingSources = null;
  var streamDone = false;

  function drainQueue() {
    if (!tokenQueue.length) {
      draining = false;
      // queue emptied — now it's safe to append sources and save history
      if (pendingSources) {
        if (msgEl) {
          msgEl.insertAdjacentHTML('beforeend', buildSourcesHtml(pendingSources.sources));
          scrollBottom();
        }
        updateDebugPanel({ sources: pendingSources.sources, debug: pendingSources.debug });
        chatHistory.push({ role: 'user', content: text });
        chatHistory.push({ role: 'assistant', content: accum });
        pendingSources = null;
      }
      if (streamDone) isTyping = false;
      return;
    }
    draining = true;
    var token = tokenQueue.shift();
    if (!msgEl) {
      hideTyping();
      const c = document.getElementById('messages');
      msgEl = document.createElement('div');
      msgEl.className = 'message bot';
      bubbleEl = document.createElement('div');
      bubbleEl.className = 'bubble';
      msgEl.appendChild(bubbleEl);
      c.appendChild(msgEl);
    }
    accum += token;
    bubbleEl.textContent = accum;
    scrollBottom();
    setTimeout(drainQueue, 18);
  }

  var folderFilter = _folderFilterValue || null;
  apiQueryStream(text, 3, currentModel, chatHistory, folderFilter, currentLang,
    function onToken(token) {
      // Strip CJK characters that leak from qwen model
      var clean = token.replace(/[\u3000-\u9fff\uf900-\ufaff\ufe30-\ufe4f\uff00-\uffef]/g, '');
      if (!clean) return;
      tokenQueue.push(clean);
      if (!draining) drainQueue();
    },
    function onSources(sources, debug) {
      pendingSources = { sources: sources, debug: debug };
    },
    function onDone(err) {
      if (err) {
        hideTyping();
        if (msgEl && bubbleEl) {
          var hint = err.partial ? 'Ответ оборвался' : 'Ошибка соединения';
          bubbleEl.textContent += '\n\n⚠ ' + hint;
        } else {
          addErrorMessage(err.partial ? 'Ответ оборвался. Попробуйте ещё раз.' : 'Нет соединения с сервером.');
        }
        isTyping = false;
        return;
      }
      streamDone = true;
      if (!draining) isTyping = false;
    }
  );
}

// ── Debug ─────────────────────────────────────────────────────────────────────

let debugVisible = false;
function toggleDebug() {
  debugVisible = !debugVisible;
  document.getElementById('debugPanel').classList.toggle('open', debugVisible);
  document.getElementById('debugToggle').classList.toggle('active', debugVisible);
}
function updateDebugPanel(data) {
  if (!data.debug) return;
  const d = data.debug;
  document.getElementById('dbgRetrieval').textContent = d.retrieval_ms || '—';
  document.getElementById('dbgGeneration').textContent = d.generation_ms || '—';
  document.getElementById('dbgTotal').textContent = d.total_ms || '—';
  document.getElementById('dbgChunks').textContent = d.chunks_after_rerank + '/' + d.chunks_retrieved;
  document.getElementById('dbgTokens').textContent = data.tokens_used || '—';
  document.getElementById('dbgModel').textContent = data.model || '';
  document.getElementById('dbgQueries').innerHTML = (d.expanded_queries || []).map(function(q, i) {
    return '<div class="debug-query-item">' + (i === 0 ? '&#8594; ' : '&#8627; ') + esc(q) + '</div>';
  }).join('');
  document.getElementById('dbgChunksList').innerHTML = (d.top_chunks || []).map(function(c) {
    const pct = Math.min(100, c.score * 100).toFixed(0);
    return '<div class="debug-chunk-item">'
      + '<span class="debug-chunk-score">' + c.score.toFixed(3) + '</span>'
      + '<div class="score-bar-wrap"><div class="score-bar" style="width:' + pct + '%"></div></div>'
      + '<span class="debug-chunk-source ' + (c.source||'') + '">' + esc(c.source || 'vec') + '</span>'
      + '<span class="debug-chunk-page">p.' + c.page_num + '</span>'
      + '<span class="debug-chunk-text">' + esc(c.text_preview) + '</span>'
      + '</div>';
  }).join('');
}

// ── PDF viewer ────────────────────────────────────────────────────────────────

if (typeof pdfjsLib !== 'undefined') {
  pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
}
let pdfDoc = null, currentPage = 1, currentDocId = null, currentHighlightText = null, isRendering = false;
let currentPdfPage = null, currentVp = null;

async function openPdfViewer(docId, page, highlightText) {
  const doc = docsData[docId] || {};
  document.getElementById('pdfPanelTitle').textContent = doc.filename || docId;
  document.getElementById('pdfPanel').classList.add('open');
  document.getElementById('pdfOverlay').classList.add('show');
  currentHighlightText = highlightText;
  document.getElementById('pdfCitation').style.display = 'none';
  if (currentDocId !== docId) {
    currentDocId = docId; pdfDoc = null;
    document.getElementById('pdfPageWrapper').style.display = 'none';
    document.getElementById('pdfLoading').style.display = 'flex';
    document.getElementById('pdfLoading').innerHTML = '<div style="font-size:22px">&#9203;</div><div>Loading...</div>';
    try {
      pdfDoc = await pdfjsLib.getDocument(getPdfUrl(docId)).promise;
      document.getElementById('pdfLoading').style.display = 'none';
      document.getElementById('pdfPageWrapper').style.display = 'inline-block';
    } catch(e) {
      document.getElementById('pdfLoading').innerHTML = '<div style="font-size:22px">&#10060;</div><div>Failed to load PDF</div>';
      return;
    }
  }
  currentPage = page || 1;
  await renderPage(currentPage);
}

async function renderPage(pageNum) {
  if (!pdfDoc || isRendering) return;
  isRendering = true;
  const page = await pdfDoc.getPage(pageNum);
  const canvas = document.getElementById('pdfCanvas');
  const textLayerDiv = document.getElementById('pdfTextLayer');
  const wrapper = document.getElementById('pdfPageWrapper');
  const container = document.getElementById('pdfCanvasContainer');
  const scale = (container.clientWidth - 28) / page.getViewport({ scale: 1 }).width;
  const vp = page.getViewport({ scale });

  canvas.width = vp.width;
  canvas.height = vp.height;
  wrapper.style.width = vp.width + 'px';
  wrapper.style.height = vp.height + 'px';

  await page.render({ canvasContext: canvas.getContext('2d'), viewport: vp }).promise;

document.getElementById('pdfPageInfo').textContent = 'Page ' + pageNum + ' of ' + pdfDoc.numPages;
  currentPdfPage = page;
  currentVp = vp;
  if (currentHighlightText) doHighlight(currentHighlightText);
  isRendering = false;
  wrapper.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function doHighlight(searchText) {
  document.querySelectorAll('.rag-highlight-box').forEach(el => el.remove());
  const citation = document.getElementById('pdfCitation');
  const citationText = document.getElementById('pdfCitationText');
  if (!searchText) { citation.style.display = 'none'; return; }

  const short = searchText.replace(/\s+/g, ' ').trim().slice(0, 160);
  citationText.textContent = short + (searchText.length > 160 ? '…' : '');
  citation.style.display = 'flex';

  if (!currentPdfPage || !currentVp) return;
  try {
    const textContent = await currentPdfPage.getTextContent();
    const items = textContent.items.filter(item => item.str);

    // Build full concatenated text + map each char → item index
    let fullText = '';
    const charMap = [];
    for (let i = 0; i < items.length; i++) {
      for (let j = 0; j < items[i].str.length; j++) charMap.push(i);
      fullText += items[i].str;
      if (items[i].hasEOL) { charMap.push(-1); fullText += ' '; }
    }

    const normalize = s => s.replace(/[\s\n]+/g, ' ').toLowerCase().trim();
    const needle = normalize(searchText).slice(0, 300);
    const haystack = normalize(fullText);
    const idx = haystack.indexOf(needle);
    if (idx === -1) return;

    const wrapper = document.getElementById('pdfPageWrapper');
    const coveredItems = new Set();
    for (let c = idx; c < idx + needle.length && c < charMap.length; c++) {
      if (charMap[c] >= 0) coveredItems.add(charMap[c]);
    }

    for (const i of coveredItems) {
      const item = items[i];
      const [a, b, c, d, x, y] = item.transform;
      const h = Math.abs(d) || Math.abs(a);
      const rect = currentVp.convertToViewportRectangle([x, y - h * 0.15, x + item.width, y + h * 0.85]);
      const x0 = Math.min(rect[0], rect[2]), y0 = Math.min(rect[1], rect[3]);
      const x1 = Math.max(rect[0], rect[2]), y1 = Math.max(rect[1], rect[3]);
      const box = document.createElement('div');
      box.className = 'rag-highlight-box';
      box.style.cssText = `left:${x0}px;top:${y0}px;width:${x1-x0}px;height:${y1-y0}px`;
      wrapper.appendChild(box);
    }

    const first = wrapper.querySelector('.rag-highlight-box');
    if (first) first.scrollIntoView({ behavior: 'smooth', block: 'center' });
  } catch(e) {
    console.warn('Highlight failed:', e);
  }
}

async function changePage(delta) {
  if (!pdfDoc) return;
  const np = currentPage + delta;
  if (np < 1 || np > pdfDoc.numPages) return;
  currentPage = np;
  await renderPage(currentPage);
}

function closePdfPanel() {
  document.getElementById('pdfPanel').classList.remove('open');
  document.getElementById('pdfOverlay').classList.remove('show');
}

// ── Init ──────────────────────────────────────────────────────────────────────

checkHealth();
loadDocuments();
loadModels();
setInterval(checkHealth, 15000);
setInterval(loadModels, 60000);
