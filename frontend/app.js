let isTyping = false;
let docsData = {};
let currentModel = null;
let availableModels = [];
// "local" (default, streamed) or "deepseek" (opt-in cloud mode — only
// selectable at all once /models reports cloud.available, which itself
// requires BOTH an administrator opt-in and a configured API key server-
// side; see rag/generator.py's GeneratorRouter). Never persisted across
// reloads on purpose — every session starts back on the safe local default.
let currentProvider = 'local';
let cloudInfo = { available: false, model: null };
let activeFolderName = '';
const openFolders = new Set();
let chatHistory = [];
let conversationStarted = false;
// Set by runCompare(), read by sendMessage()'s follow-ups, cleared on
// backToWelcome() — without this, "Compare A and B" correctly scoped its
// own answer to just those documents, but the very next follow-up
// ("Which one has the later effective date?") went through the ordinary
// sendMessage() path with no documentIds at all and searched the whole
// active folder again, silently dropping the scope a user would
// reasonably expect a comparison conversation to keep.
let activeDocumentIds = null;
let debugVisible = false;
let docSearchQuery = '';

// Comparing every document in a large folder used to list all N filenames in
// the question text and set top_k = N*4 uncapped — for a 150-doc folder
// that's top_k=600, which retrieve_expanded() turns into a ~3000-candidate
// rerank pass (max(20, top_k*5)) despite the prompt only ever using the
// first ~3000 tokens of context anyway (see MAX_CONTEXT_CHARS in
// rag/prompt_builder.py). The API now hard-rejects top_k>20 regardless
// (QueryRequest.top_k, api/main.py), but the real fix is not asking for it
// in the first place: cap how many documents a single compare can name, and
// make the user actually choose which ones above that cap.
const MAX_COMPARE_DOCS = 5;
var compareSelection = null; // { docs: [...], selected: Set<doc_id> } while picking

// Real, verified questions against the actual corpus — not abstract
// placeholders — so a first click reliably lands on a convincing answer
// instead of a hedge. Each was spot-checked live against /query/stream
// before being added here.
const SUGGESTIONS = [
  { icon: 'clock', text: 'When does the Energy Performance of Buildings (England and Wales) Regulations 2012 come into force?' },
  { icon: 'file-text', text: 'What is the maximum discount limit under the Housing (Right to Buy) (Limit on Discount) (England) Order 2012?' },
  { icon: 'search', text: 'What is the case Jintu Das vs The State of Assam about?' },
  { icon: 'columns', text: 'Compare the Localism Act 2011 Commencement No. 6 and Commencement No. 8 orders — what changed?' },
];

// Escapes for BOTH text content and attribute-value interpolation. The old
// implementation (a textContent -> innerHTML round-trip) only escaped
// &/</> — safe for text nodes, but quotes aren't special there, so it left
// '"' and "'" untouched. Every call site below embeds this inside a
// double-quoted HTML attribute (data-fname="...", title="..."), so a folder
// or document name containing either character broke the markup outright
// — e.g. a folder named Client's docs closed an inline-JS string literal
// early (see the folder-tree onclick rewrite below), and a name with a
// literal '"' would have closed an attribute value early even after that.
function esc(text) {
  return String(text == null ? '' : text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}
function scrollBottom() { const m = document.getElementById('messages'); m.scrollTop = m.scrollHeight; }

// ── Icon injection (static, one-time) ───────────────────────────────────────

function isAdminMode() {
  // Evaluation is an engineering/admin tool, not something a regular user
  // of the knowledge base needs — hidden from the topbar by default.
  // ?admin=1 once (persisted to localStorage) keeps it visible after.
  if (new URLSearchParams(window.location.search).has('admin')) {
    localStorage.setItem('kb_admin', '1');
  }
  return localStorage.getItem('kb_admin') === '1';
}

function injectStaticIcons() {
  document.getElementById('mobileSidebarToggle').innerHTML = svgIcon('menu', 18);
  document.getElementById('brandMark').innerHTML = svgIcon('layers', 16);
  if (isAdminMode()) {
    const evalLink = document.getElementById('evalLink');
    evalLink.innerHTML = svgIcon('bar-chart', 15) + '<span>Evaluation</span>';
    evalLink.style.display = '';
  }
  document.getElementById('settingsBtn').innerHTML = svgIcon('settings', 17);
  document.getElementById('apiKeyRow').innerHTML = svgIcon('key', 15) + '<span>API key</span>';
  document.getElementById('addDocsBtn').innerHTML = svgIcon('upload', 14) + '<span>Add documents</span>';
  document.getElementById('addDocsFilesItem').innerHTML = svgIcon('file', 14) + '<span>Upload documents</span>'
    + '<input type="file" accept=".pdf,.txt" id="fileInput" multiple onchange="uploadFiles(this)">';
  document.getElementById('addDocsFolderItem').innerHTML = svgIcon('folder', 14) + '<span>Upload folder</span>'
    + '<input type="file" id="folderFileInput" webkitdirectory multiple onchange="uploadFolder(this)">';
  document.getElementById('docSearchBox').insertAdjacentHTML('afterbegin', svgIcon('search', 14));
  document.getElementById('welcomeSendBtn').innerHTML = svgIcon('send', 15);
  document.getElementById('chatSendBtn').innerHTML = svgIcon('send', 15);
  document.getElementById('retrievalToggle').innerHTML = svgIcon('activity', 13) + '<span>Retrieval details</span>';
  document.getElementById('retrievalTitleIcon').innerHTML = svgIcon('activity', 12) + ' Retrieval details';
  document.getElementById('retrievalCloseBtn').innerHTML = svgIcon('x', 13);
  document.getElementById('folderFilterBtn').querySelector('.chevron').innerHTML = svgIcon('chevron-down', 11);
  document.getElementById('welcomeScopeBtn').querySelector('.chevron').innerHTML = svgIcon('chevron-down', 11);
  document.getElementById('pdfPanelIcon').innerHTML = svgIcon('file-text', 14);
  document.getElementById('pdfPanelCloseBtn').innerHTML = svgIcon('x', 14);
  document.getElementById('pdfPrevBtn').innerHTML = svgIcon('chevron-left', 14);
  document.getElementById('pdfNextBtn').innerHTML = svgIcon('chevron-right', 14);
  document.getElementById('pdfViewTextBtn').innerHTML = svgIcon('file-text', 13) + '<span>View source text</span>';
  document.getElementById('pdfLoadingIcon').innerHTML = svgIcon('file-text', 26);
  document.getElementById('txtPanelIcon').innerHTML = svgIcon('file-text', 14);
  document.getElementById('txtPanelCloseBtn').innerHTML = svgIcon('x', 14);
  document.getElementById('txtPrevBtn').innerHTML = svgIcon('chevron-left', 14);
  document.getElementById('txtNextBtn').innerHTML = svgIcon('chevron-right', 14);
  document.getElementById('txtViewOriginalBtn').innerHTML = svgIcon('external-link', 13) + '<span>View original PDF</span>';

  const grid = document.getElementById('suggestionGrid');
  grid.innerHTML = SUGGESTIONS.map(function(s) {
    return '<button class="suggestion-card" onclick="useSuggestion(this)">' + svgIcon(s.icon, 16)
      + '<span class="suggestion-text">' + esc(s.text) + '</span></button>';
  }).join('');
}

// ── Health ────────────────────────────────────────────────────────────────────

async function checkHealth() {
  try {
    const ok = await apiHealth();
    document.getElementById('statusDot').classList.toggle('err', !ok);
    document.getElementById('statusText').textContent = ok ? 'online' : 'error';
  } catch(e) {
    document.getElementById('statusDot').classList.add('err');
    document.getElementById('statusText').textContent = 'offline';
  }
}

// ── Models (now inside the settings panel) ──────────────────────────────────

async function loadModels() {
  try {
    const data = await apiGetModels();
    availableModels = data.models || [];
    // Only adopt the server's "current" on first load — after that, an
    // explicit selectModel() choice must win over the periodic poll (the
    // backend has no persistent model-switch state, /models always reports
    // its .env default, which would otherwise silently revert the user's pick).
    if (currentModel === null) currentModel = data.current;
    cloudInfo = data.cloud || { available: false, model: null };
    // If the cloud provider became unavailable mid-session (admin disabled
    // it, key removed) after the user had selected it, fall back to local
    // rather than silently continuing to claim "deepseek" on the next
    // request — GeneratorRouter would reject it anyway, but this avoids a
    // confusing error appearing with no visible cause in the UI.
    if (currentProvider === 'deepseek' && !cloudInfo.available) currentProvider = 'local';
    renderModelList();
    renderProviderList();
  } catch(e) {
    document.getElementById('modelList').innerHTML = '<div class="model-loading">Unavailable</div>';
  }
}

function renderModelList() {
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
    html += '<span class="model-option-check">' + svgIcon('check', 13) + '</span></div>';
  });
  list.innerHTML = html;
}

function selectModel(name) {
  currentModel = name;
  renderModelList();
}

// ── Provider (local Qwen vs. opt-in DeepSeek cloud mode) ─────────────────────
// Section is hidden entirely (see renderProviderList) unless the server
// reports the cloud generator as available — no disabled/greyed-out toggle
// cluttering the default, local-only experience.

function renderProviderList() {
  const section = document.getElementById('providerSection');
  if (!cloudInfo.available) { section.style.display = 'none'; return; }
  section.style.display = '';
  const options = [
    { id: 'local', label: 'Local (Qwen)', hint: 'Never leaves this server' },
    { id: 'deepseek', label: 'DeepSeek (cloud)', hint: cloudInfo.model || 'deepseek' },
  ];
  let html = '';
  options.forEach(function(o) {
    html += '<div class="model-option ' + (o.id === currentProvider ? 'active' : '') + '" onclick="selectProvider(' + JSON.stringify(o.id).replace(/"/g, "'") + ')">';
    html += '<div class="model-option-name">' + esc(o.label) + '</div>';
    html += '<div class="model-option-size">' + esc(o.hint) + '</div>';
    html += '<span class="model-option-check">' + svgIcon('check', 13) + '</span></div>';
  });
  document.getElementById('providerList').innerHTML = html;

  // Persistent, impossible-to-miss warning while DeepSeek is the active
  // choice — not just a passive hint next to the option, since the user
  // asked specifically for something visible "when you select it", in the
  // same style already used elsewhere for flagged findings (.finding-open,
  // the warning color/badge shape from the evaluation page).
  const warning = document.getElementById('providerCloudWarning');
  if (currentProvider === 'deepseek') {
    warning.style.display = '';
    warning.innerHTML = svgIcon('alert-triangle', 13) + ' Document content for this chat is sent to DeepSeek’s API (external service)';
  } else {
    warning.style.display = 'none';
  }
}

function selectProvider(id) {
  currentProvider = id;
  renderProviderList();
}

function toggleSettings() {
  const p = document.getElementById('settingsPanel');
  p.classList.toggle('show');
}

// Below 860px the sidebar (folders, upload, document search) has no other
// entry point — the CSS just hid it outright with no way back in, which
// meant uploading or picking a document was impossible on a phone/narrow
// tablet. Slides it in as an overlay drawer instead.
function toggleMobileSidebar(force) {
  const sidebar = document.getElementById('sidebarEl');
  const overlay = document.getElementById('mobileSidebarOverlay');
  const open = force !== undefined ? force : !sidebar.classList.contains('mobile-open');
  sidebar.classList.toggle('mobile-open', open);
  overlay.classList.toggle('show', open);
}

document.getElementById('retrievalSwitch') && document.getElementById('retrievalSwitch').addEventListener('click', function(e){ e.stopPropagation(); });

// Dynamically-rendered rows (folders, docs, sources, recent) use role="button"
// on a <div> rather than a native <button> (they carry drag-and-drop and
// data-* attributes a button complicates) — this is the one place that needs
// to make Enter/Space activate them the way a real button gets for free.
document.addEventListener('keydown', function(e) {
  if (e.key !== 'Enter' && e.key !== ' ') return;
  var el = e.target.closest('[role="button"]');
  if (!el) return;
  e.preventDefault();
  el.click();
});

document.addEventListener('click', function(e) {
  // Settings panel close
  var sw = document.getElementById('settingsWrap');
  if (sw && !sw.contains(e.target)) document.getElementById('settingsPanel').classList.remove('show');

  // Add-docs menu close
  var adw = document.getElementById('addDocsWrap');
  if (adw && !adw.contains(e.target)) document.getElementById('addDocsMenu').classList.remove('show');

  // Source card click
  var srcEl = e.target.closest('[data-src]');
  if (srcEl) {
    var s = _sourcesStore[parseInt(srcEl.dataset.src, 10)];
    if (s) {
      openTextViewer(s.document, s.page || 1, s.char_start, s.char_end);
    }
  }

  // Scope dropdown close-on-outside-click / option select — shared by both
  // the input-bar's folder-filter-btn and the welcome screen's own picker.
  [{ dd: 'folderFilterDropdown', wrap: 'folderFilterWrap' }, { dd: 'welcomeScopeDropdown', wrap: 'welcomeScopeWrap' }].forEach(function(pair) {
    var dd = document.getElementById(pair.dd);
    var wrap = document.getElementById(pair.wrap);
    if (dd && wrap && dd.classList.contains('open') && !wrap.contains(e.target)) {
      dd.classList.remove('open');
      updateFolderFilterSelect();
    }
  });
  var ffOpt = e.target.closest('[data-ff]');
  if (ffOpt && ffOpt.closest('.scope-dropdown')) {
    _folderFilterValue = ffOpt.dataset.ff;
    activeFolderName = ffOpt.dataset.ff; // keep the sidebar highlight in sync with the actual search scope
    activeDocumentIds = null; // explicitly picking a folder/all-folders overrides a sticky compare scope
    ffOpt.closest('.scope-dropdown').classList.remove('open');
    updateFolderFilterSelect();
    renderDocTree();
  }
});

function toggleAddDocsMenu(e) {
  e.stopPropagation();
  document.getElementById('addDocsMenu').classList.toggle('show');
}

// ── Folder tree ───────────────────────────────────────────────────────────────

function getFolderMap() {
  const map = {};
  Object.values(docsData).forEach(function(doc) {
    const f = doc.folder || 'Uncategorized';
    doc.folder = f;
    if (!map[f]) map[f] = [];
    if (!doc._placeholder) map[f].push(doc);
  });
  Object.values(docsData).forEach(function(doc) {
    if (doc._placeholder) {
      const f = doc.folder;
      if (!map[f]) map[f] = [];
    }
  });
  return map;
}

function filterDocTree(q) {
  docSearchQuery = (q || '').trim().toLowerCase();
  renderDocTree();
}

function renderDocTree() {
  if (typeof updateFolderFilterSelect === 'function') updateFolderFilterSelect();
  const list = document.getElementById('docsList');
  const realDocs = Object.values(docsData).filter(function(d){ return !d._placeholder; });
  const total = realDocs.length;

  if (Object.keys(docsData).length === 0) {
    list.innerHTML = '<div class="empty-docs">No documents yet<br><span style="font-size:11px">Add documents to start</span></div>';
    document.getElementById('workspaceMeta').textContent = 'No documents';
    return;
  }

  const map = getFolderMap();
  let names = Object.keys(map).sort(function(a, b) {
    if (a === 'Uncategorized') return 1;
    if (b === 'Uncategorized') return -1;
    return a.localeCompare(b);
  });

  if (docSearchQuery) {
    const filteredMap = {};
    names.forEach(function(fname) {
      const matches = (map[fname] || []).filter(function(d) { return (d.filename || '').toLowerCase().indexOf(docSearchQuery) !== -1; });
      if (matches.length || fname.toLowerCase().indexOf(docSearchQuery) !== -1) filteredMap[fname] = matches;
    });
    names = Object.keys(filteredMap);
    if (!names.length) {
      list.innerHTML = '<div class="empty-docs">No documents match &ldquo;' + esc(docSearchQuery) + '&rdquo;</div>';
      return;
    }
  }

  let html = '';
  names.forEach(function(fname) {
    const docs = (docSearchQuery ? (map[fname] || []).filter(function(d){ return (d.filename||'').toLowerCase().indexOf(docSearchQuery) !== -1; }) : map[fname]) || [];
    const isActive = fname === activeFolderName;
    const isOpen = openFolders.has(fname) || !!docSearchQuery;
    const isUncategorized = fname === 'Uncategorized';

    html += '<div class="folder-group' + (isActive ? ' active' : '') + '">';

    // data-fname (read via event delegation below) instead of building an
    // onclick="fn('...')" string with the name spliced in — a name
    // containing a quote character used to break the generated JS/HTML
    // outright (see esc() comment above).
    html += '<div class="folder-header" role="button" tabindex="0" data-fname="' + esc(fname) + '">';
    html += '<span class="folder-arrow' + (isOpen ? ' open' : '') + '">' + svgIcon('chevron-right', 12) + '</span>';
    html += '<span class="folder-icon">' + svgIcon(isOpen ? 'folder-open' : 'folder', 15) + '</span>';
    html += '<span class="folder-name" title="' + esc(fname) + '">' + esc(fname) + '</span>';
    html += '<span class="folder-count">' + docs.length + '</span>';
    html += '<div class="folder-actions">';
    if (docs.length >= 2) {
      html += '<button class="fld-btn fld-compare" title="Compare documents in this folder">' + svgIcon('columns', 13) + '</button>';
    }
    html += '<button class="fld-btn fld-upload" title="Upload documents here">' + svgIcon('upload', 13) + '</button>';
    if (!isUncategorized) {
      html += '<button class="fld-btn fld-rename" title="Rename">' + svgIcon('edit', 13) + '</button>';
      html += '<button class="fld-btn del fld-delete" title="Delete">' + svgIcon('trash', 13) + '</button>';
    }
    html += '</div></div>';

    if (isOpen) {
      html += '<div class="folder-files">';
      if (docs.length === 0) {
        html += '<div style="font-size:11.5px;color:var(--text-muted);padding:7px 8px;">Empty — upload documents here</div>';
      } else {
        docs.forEach(function(doc) {
          var sel = (compareSelection && compareSelection.eligibleIds.has(doc.doc_id)) ? compareSelection : null;
          html += renderDocItem(doc, sel);
        });
      }
      html += '</div>';
    }
    html += '</div>';
  });

  if (!docSearchQuery) html += '<button class="new-folder-btn">' + svgIcon('plus', 13) + ' New folder</button>';
  list.innerHTML = html;

  var newFolderBtn = list.querySelector('.new-folder-btn');
  if (newFolderBtn) newFolderBtn.addEventListener('click', createNewFolder);

  list.querySelectorAll('.folder-header').forEach(function(header) {
    var fname = header.dataset.fname; // decoded back to the raw name by the DOM, entities and all
    header.addEventListener('click', function() { toggleFolder(fname); });
    header.addEventListener('dragover', function(e) { e.preventDefault(); header.parentElement.style.outline = '1px solid var(--accent)'; });
    header.addEventListener('dragleave', function() { header.parentElement.style.outline = ''; });
    header.addEventListener('drop', function(e) { dropOnFolder(e, fname); });
    var compareBtn = header.querySelector('.fld-compare');
    if (compareBtn) compareBtn.addEventListener('click', function(e) { e.stopPropagation(); compareInFolder(fname); });
    var uploadBtn = header.querySelector('.fld-upload');
    if (uploadBtn) uploadBtn.addEventListener('click', function(e) { e.stopPropagation(); uploadToFolder(fname); });
    var renameBtn = header.querySelector('.fld-rename');
    if (renameBtn) renameBtn.addEventListener('click', function(e) { e.stopPropagation(); renameFolder(fname); });
    var deleteBtn = header.querySelector('.fld-delete');
    if (deleteBtn) deleteBtn.addEventListener('click', function(e) { confirmDeleteFolder(e, fname); });
  });

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
      if (!doc) return;
      if (compareSelection && compareSelection.eligibleIds.has(item.dataset.docId)) {
        toggleCompareDoc(item.dataset.docId);
        return;
      }
      openTextViewer(item.dataset.docId, 1, null, null);
    });
  });

  const totalChunks = realDocs.reduce(function(s, d){ return s + (d.chunks || d.chunks_created || 0); }, 0);
  document.getElementById('workspaceMeta').innerHTML = total + ' document' + (total===1?'':'s') +
    ' <span title="' + totalChunks + ' indexed passages">&middot; ' + totalChunks + ' passages</span>';
  renderCompareBar();
}

function renderDocItem(doc, selection) {
  const chunks = doc.chunks_created || doc.chunks || 0;
  const pages = doc.pages || 0;
  const inSelection = !!selection;
  const checked = inSelection && selection.selected.has(doc.doc_id);
  const leadIcon = inSelection
    ? '<span class="doc-check' + (checked ? ' checked' : '') + '">' + (checked ? svgIcon('check', 12) : '') + '</span>'
    : '<span class="doc-icon">' + svgIcon('file-text', 14) + '</span>';
  return '<div class="doc-item' + (checked ? ' selected' : '') + '" data-doc-id="' + doc.doc_id + '" role="button" tabindex="0"'
    + ' draggable="' + (!inSelection) + '"'
    + ' ondragstart="event.dataTransfer.setData(\'docId\',\'' + doc.doc_id + '\');this.style.opacity=\'0.4\'"'
    + ' ondragend="this.style.opacity=\'1\'">'
    + leadIcon
    + '<div class="doc-info">'
    + '<div class="doc-name" title="' + esc(doc.filename) + '">' + esc(doc.filename) + '</div>'
    + '<div class="doc-meta">' + pages + 'p &middot; ' + chunks + ' chunks</div>'
    + '</div>'
    + (inSelection ? '' : '<button class="doc-delete" data-doc-id="' + doc.doc_id + '" title="Delete">' + svgIcon('x', 13) + '</button>')
    + '</div>';
}

// ── Folder actions ────────────────────────────────────────────────────────────

function toggleFolder(fname) {
  if (openFolders.has(fname)) {
    openFolders.delete(fname);
  } else {
    openFolders.add(fname);
    // "Uncategorized" isn't a real folder server-side (it's this frontend's
    // label for documents with no folder at all), so there's no filter
    // value to scope a search to it by — setActiveFolder() would highlight
    // it as "selected" while _folderFilterValue silently stayed empty
    // (searching everything), which looked like a real, working filter and
    // wasn't. Expandable to browse, but not selectable as a search scope.
    if (fname !== 'Uncategorized') setActiveFolder(fname);
  }
  renderDocTree();
}

function uploadToFolder(fname) {
  setActiveFolder(fname);
  document.getElementById('fileInput').click();
}

function setActiveFolder(fname) {
  // A sticky compare scope (see activeDocumentIds's declaration comment)
  // only gets cleared by backToWelcome() or explicitly picking a folder
  // from the input-bar's own dropdown — clicking a folder directly in the
  // sidebar went through this function instead and left activeDocumentIds
  // untouched, so the next question could carry both a new folder *and*
  // stale document_ids from an earlier compare, sometimes scoping to an
  // intersection that's empty (a folder that doesn't contain those
  // documents) and coming back with nothing.
  activeDocumentIds = null;
  activeFolderName = fname;
  // The sidebar selection IS the search scope by default — without this,
  // picking a folder in the sidebar looked like it scoped the next question
  // (folder-header highlights, "active" state) but silently didn't, since
  // runQuery() actually reads _folderFilterValue, a separate variable only
  // otherwise set via the input-bar's own folder-filter dropdown.
  _folderFilterValue = (fname && fname !== 'Uncategorized') ? fname : '';
  updateFolderFilterSelect();
  renderDocTree();
}

function clearActiveFolder() {
  activeDocumentIds = null;
  activeFolderName = '';
  _folderFilterValue = '';
  updateFolderFilterSelect();
  renderDocTree();
}

async function createNewFolder() {
  const name = prompt('Folder name:');
  if (!name || !name.trim()) return;
  const fname = name.trim();
  // Wait for server confirmation before touching local state — previously
  // this added the placeholder and moved on immediately, so a failed
  // request (name clash, connection drop) left a folder in the sidebar
  // that never actually existed server-side.
  try {
    const r = await fetch(API + '/folders', { method: 'POST', headers: authHeaders({'Content-Type':'application/json'}), body: JSON.stringify({name: fname}) });
    if (!r.ok) { alert('Could not create folder "' + fname + '".'); return; }
  } catch (e) {
    alert('Could not create folder — connection error.');
    return;
  }
  docsData['__ph__' + fname] = { doc_id: '__ph__' + fname, filename: '', folder: fname, _placeholder: true, pages: 0, chunks: 0 };
  openFolders.add(fname);
  setActiveFolder(fname);
}

async function renameFolder(oldName) {
  const newName = prompt('New name:', oldName);
  if (!newName || !newName.trim() || newName.trim() === oldName) return;
  const n = newName.trim();
  try {
    const r = await fetch(API + '/folders/' + encodeURIComponent(oldName), { method: 'PATCH', headers: authHeaders({'Content-Type':'application/json'}), body: JSON.stringify({name: n}) });
    if (!r.ok) { alert('Could not rename folder.'); return; }
  } catch (e) {
    alert('Could not rename folder — connection error.');
    return;
  }
  Object.values(docsData).forEach(function(doc) {
    if (doc.folder === oldName) {
      doc.folder = n;
      if (doc._placeholder) { doc.doc_id = '__ph__' + n; delete docsData['__ph__' + oldName]; docsData['__ph__' + n] = doc; }
    }
  });
  if (openFolders.has(oldName)) { openFolders.delete(oldName); openFolders.add(n); }
  if (activeFolderName === oldName) setActiveFolder(n); else renderDocTree();
}

async function dropOnFolder(event, targetFolder) {
  event.preventDefault();
  event.currentTarget.parentElement.style.outline = '';
  const docId = event.dataTransfer.getData('docId');
  if (!docId || !docsData[docId]) return;
  const previousFolder = docsData[docId].folder;
  if (previousFolder === targetFolder) return;
  docsData[docId].folder = targetFolder;
  renderDocTree();
  let ok = false;
  try {
    ok = await apiUpdateFolder(docId, targetFolder);
  } catch (e) { ok = false; }
  if (!ok) {
    // Roll back — the sidebar had already jumped the document to
    // targetFolder optimistically, and previously just stayed there even
    // if the server rejected the move (or the request failed outright),
    // silently diverging from what's actually persisted.
    if (docsData[docId]) docsData[docId].folder = previousFolder;
    alert('Could not move the document — please try again.');
    renderDocTree();
  }
}

let _deletePending = null;

function confirmDeleteFolder(event, fname) {
  event.stopPropagation();
  const btn = event.currentTarget;
  if (_deletePending === fname) {
    _deletePending = null;
    deleteFolder(fname);
    return;
  }
  _deletePending = fname;
  btn.textContent = 'Delete?';
  btn.classList.add('del-confirm');
  setTimeout(function() {
    if (_deletePending === fname) {
      _deletePending = null;
      renderDocTree();
    }
  }, 3000);
}

async function deleteFolder(fname) {
  const docs = Object.values(docsData).filter(function(d){ return d.folder === fname && !d._placeholder; });
  // Wait for every document delete to actually finish (each already removes
  // itself from docsData on success — see deleteDocument()) before deciding
  // the folder is empty. Firing these without awaiting and immediately
  // wiping the folder's local entries meant a partial failure left the UI
  // showing "folder gone" while the server still had some of its documents.
  const results = await Promise.all(docs.map(function(doc) { return deleteDocument(doc.doc_id, doc.filename, true); }));
  const failed = results.filter(function(ok) { return !ok; }).length;
  if (failed > 0) {
    alert(failed + ' of ' + docs.length + ' document(s) in "' + fname + '" could not be deleted. The folder was kept — please retry.');
    renderDocTree();
    return;
  }
  let folderDeleteOk = false;
  try {
    const r = await fetch(API + '/folders/' + encodeURIComponent(fname), { method: 'DELETE', headers: authHeaders() });
    folderDeleteOk = r.ok;
  } catch (e) {
    folderDeleteOk = false;
  }
  if (!folderDeleteOk) {
    // The folder registration still exists server-side even though every
    // document in it is now gone — previously the code deleted the local
    // placeholder and hid the folder regardless, so it looked removed
    // until the next loadDocuments() (page reload, key change, ...)
    // re-fetched it from the server and it reappeared, now empty, with no
    // indication anything had gone wrong the first time.
    alert('All documents were deleted, but the empty folder itself could not be removed — it will stay listed (now empty). Try deleting it again.');
    docsData['__ph__' + fname] = { doc_id: '__ph__' + fname, filename: '', folder: fname, _placeholder: true, pages: 0, chunks: 0 };
    renderDocTree();
    return;
  }
  delete docsData['__ph__' + fname]; // an empty folder is represented locally only by this placeholder
  openFolders.delete(fname);
  if (activeFolderName === fname) clearActiveFolder(); else renderDocTree();
}

// ── Documents ─────────────────────────────────────────────────────────────────

var _folderFilterValue = '';

// Builds the same "All folders / <folder list>" option list for both scope
// pickers — the persistent input-bar's folder-filter-btn (visible once a
// conversation has started) and the welcome screen's own dropdown (added
// so picking a folder doesn't require already knowing the sidebar does
// this too — the welcome screen previously only showed the *result* of a
// sidebar click as inert text, with no way to change it from there).
function updateFolderFilterSelect() {
  var folders = Object.keys(getFolderMap()).filter(function(f){ return f !== 'Uncategorized'; }).sort();
  if (_folderFilterValue && folders.indexOf(_folderFilterValue) === -1) {
    _folderFilterValue = '';
  }
  var html = '<div class="ff-dropdown-header">Search in</div>';
  html += '<div class="ff-option' + (!_folderFilterValue ? ' selected' : '') + '" data-ff=""><span>All folders</span><span class="ff-check">' + svgIcon('check', 12) + '</span></div>';
  folders.forEach(function(f) {
    html += '<div class="ff-option' + (_folderFilterValue === f ? ' selected' : '') + '" data-ff="' + esc(f) + '"><span>' + esc(f) + '</span><span class="ff-check">' + svgIcon('check', 12) + '</span></div>';
  });
  ['folderFilterDropdown', 'welcomeScopeDropdown'].forEach(function(id) {
    var dd = document.getElementById(id);
    if (dd) dd.innerHTML = html;
  });

  // While a compare scope is active, that's the real enforced filter
  // (document_ids, strictly narrower than the folder) — show it instead of
  // the folder name so a sticky scope the user didn't explicitly ask to
  // keep isn't silently invisible on every follow-up.
  var scopeText = activeDocumentIds ? 'Comparing ' + activeDocumentIds.length + ' docs' : (_folderFilterValue || 'All folders');

  var ffLabel = document.getElementById('folderFilterLabel');
  if (ffLabel) ffLabel.textContent = scopeText;
  var ffBtn = document.getElementById('folderFilterBtn');
  if (ffBtn) {
    var ffOpen = document.getElementById('folderFilterDropdown').classList.contains('open');
    ffBtn.classList.toggle('active', !!_folderFilterValue || !!activeDocumentIds || ffOpen);
    var ffChev = ffBtn.querySelector('.chevron');
    if (ffChev && !ffChev.innerHTML) ffChev.innerHTML = svgIcon('chevron-down', 11);
  }

  var welcomeVal = document.getElementById('searchScopeWelcomeValue');
  if (welcomeVal) welcomeVal.textContent = activeDocumentIds ? scopeText : (_folderFilterValue || 'All legal documents');
  var welcomeBtn = document.getElementById('welcomeScopeBtn');
  if (welcomeBtn) {
    var wOpen = document.getElementById('welcomeScopeDropdown').classList.contains('open');
    welcomeBtn.classList.toggle('active', !!_folderFilterValue || !!activeDocumentIds || wOpen);
    var wChev = welcomeBtn.querySelector('.chevron');
    if (wChev && !wChev.innerHTML) wChev.innerHTML = svgIcon('chevron-down', 11);
  }
}

// Shared by both scope pickers — dropdownId is whichever one this button
// owns. Closes the other one first so at most one is ever open, same as
// any other single-open-at-a-time menu on this page.
function toggleScopeDropdown(e, dropdownId) {
  e.stopPropagation();
  var dd = document.getElementById(dropdownId);
  var opening = !dd.classList.contains('open');
  ['folderFilterDropdown', 'welcomeScopeDropdown'].forEach(function(id) {
    if (id !== dropdownId) { var other = document.getElementById(id); if (other) other.classList.remove('open'); }
  });
  dd.classList.toggle('open', opening);
  updateFolderFilterSelect();
}

async function loadDocuments() {
  try {
    const data = await apiGetDocuments();
    docsData = {};
    (data.documents || []).forEach(function(doc) { docsData[doc.doc_id] = doc; });
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

// Kept in one place — api/main.py's PARSERS_BY_EXT (pdf, txt) is the actual
// source of truth for what the backend accepts; this just needs to stay in
// sync with it so the upload picker doesn't silently drop a format the
// server would otherwise happily ingest.
function isSupportedDocFile(filename) {
  var lower = filename.toLowerCase();
  return lower.endsWith('.pdf') || lower.endsWith('.txt');
}

async function uploadFiles(input) {
  document.getElementById('addDocsMenu').classList.remove('show');
  const files = Array.from(input.files).filter(function(f){ return isSupportedDocFile(f.name); });
  if (!files.length) return;
  const folder = (activeFolderName && activeFolderName !== 'Uncategorized') ? activeFolderName : '';
  showProg();

  if (files.length === 1) {
    setProg(30, 'Uploading…');
    try {
      setProg(60, 'Processing…');
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
        document.getElementById('progressFill').style.background = 'var(--warning)';
      } else {
        setProg(100, 'Upload failed');
        document.getElementById('progressFill').style.background = 'var(--danger)';
      }
    } catch(e) { setProg(100, 'Error'); document.getElementById('progressFill').style.background = 'var(--danger)'; }
  } else {
    setProg(20, 'Uploading ' + files.length + ' files…');
    try {
      setProg(60, 'Processing…');
      const res = await apiUploadBatch(files, folder);
      if (res.ok) {
        const data = res.data;
        data.results.forEach(function(r) {
          if (r.status === 'indexed') { r.folder = folder || 'Uncategorized'; docsData[r.doc_id] = r; }
        });
        if (folder) { const ph = '__ph__' + folder; if (docsData[ph]) delete docsData[ph]; }
        setProg(100, 'Done: ' + data.indexed + ' indexed' + (data.skipped ? ', ' + data.skipped + ' skipped' : ''));
        renderDocTree();
      } else { setProg(100, 'Failed'); document.getElementById('progressFill').style.background = 'var(--danger)'; }
    } catch(e) { setProg(100, 'Error'); document.getElementById('progressFill').style.background = 'var(--danger)'; }
  }
  hideProg(2000);
  input.value = '';
}

async function uploadFolder(input) {
  document.getElementById('addDocsMenu').classList.remove('show');
  const docFiles = Array.from(input.files).filter(function(f){ return isSupportedDocFile(f.name); });
  if (!docFiles.length) { alert('No supported documents (.pdf, .txt) found in folder'); input.value = ''; return; }

  showProg();
  const byFolder = {};
  docFiles.forEach(function(f) {
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
    setProg(Math.round(done / folderList.length * 90) + 5, fname + ' (' + byFolder[fname].length + ')…');
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

  setProg(100, indexed + ' indexed' + (skipped ? ', ' + skipped + ' skipped' : ''));
  renderDocTree();
  hideProg(3000);
  input.value = '';
}

async function deleteDocument(doc_id, filename, silent) {
  if (!silent && !confirm('Delete "' + filename + '"?')) return false;
  try {
    const ok = await apiDeleteDocument(doc_id);
    if (ok) { delete docsData[doc_id]; if (!silent) renderDocTree(); }
    return ok;
  } catch(e) {
    if (!silent) alert('Connection error');
    return false;
  }
}

function compareInFolder(fname) {
  setActiveFolder(fname);
  compareDocuments();
}

function compareDocuments() {
  var allDocs = Object.values(docsData).filter(function(d){ return !d._placeholder && d.filename; });
  var scopedDocs = activeFolderName
    ? allDocs.filter(function(d){ return d.folder === activeFolderName; })
    : allDocs;

  if (scopedDocs.length < 2) return;

  if (scopedDocs.length > MAX_COMPARE_DOCS) {
    startCompareSelection(scopedDocs);
    return;
  }
  runCompare(scopedDocs);
}

function runCompare(docs) {
  var names = docs.map(function(d){ return d.filename; }).join(', ');
  var folderCtx = activeFolderName ? ' (folder: ' + activeFolderName + ')' : '';

  _folderFilterValue = activeFolderName || '';
  updateFolderFilterSelect();

  var text = 'Compare these documents in detail' + folderCtx + ': ' + names + '. ' +
    'For each document provide: 1) main subject and key arguments, 2) parties or entities involved, 3) conclusions or outcomes. ' +
    'Then compare them: what are the key differences and similarities? Be thorough and specific.';

  // The filenames spelled out above are context for the LLM's answer, not
  // the actual retrieval scope — without document_ids, hybrid search (and
  // the reranker) stayed free to pull in *other* documents from the same
  // folder, so "compare these 3" could silently answer about a 4th. See
  // rag/retriever.py::retrieve_expanded's document_ids param.
  var documentIds = docs.map(function(d){ return d.doc_id; });
  activeDocumentIds = documentIds; // sticky for this conversation's follow-ups — see declaration comment

  enterConversationMode();
  runQuery(text, Math.min(docs.length * 4, 20), documentIds);
}

// ── Compare-document picker (folders larger than MAX_COMPARE_DOCS) ─────────

function startCompareSelection(docs) {
  compareSelection = { docs: docs, eligibleIds: new Set(docs.map(function(d){ return d.doc_id; })), selected: new Set() };
  if (activeFolderName) openFolders.add(activeFolderName);
  renderDocTree();
}

function cancelCompareSelection() {
  compareSelection = null;
  renderDocTree();
}

function toggleCompareDoc(docId) {
  if (!compareSelection || !compareSelection.eligibleIds.has(docId)) return;
  if (compareSelection.selected.has(docId)) {
    compareSelection.selected.delete(docId);
  } else if (compareSelection.selected.size < MAX_COMPARE_DOCS) {
    compareSelection.selected.add(docId);
  }
  renderDocTree();
}

function confirmCompareSelection() {
  if (!compareSelection || compareSelection.selected.size < 2) return;
  var chosen = compareSelection.docs.filter(function(d) { return compareSelection.selected.has(d.doc_id); });
  compareSelection = null;
  renderDocTree();
  runCompare(chosen);
}

function renderCompareBar() {
  var existing = document.getElementById('compareBar');
  if (existing) existing.remove();
  if (!compareSelection) return;
  var n = compareSelection.selected.size;
  var bar = document.createElement('div');
  bar.id = 'compareBar';
  bar.className = 'compare-bar';
  bar.innerHTML =
    '<span>' + svgIcon('columns', 14) + ' Pick 2–' + MAX_COMPARE_DOCS + ' documents to compare (' + n + ' selected)</span>' +
    '<button class="compare-bar-btn" ' + (n < 2 ? 'disabled' : '') + ' onclick="confirmCompareSelection()">Compare ' + n + '</button>' +
    '<button class="compare-bar-cancel" onclick="cancelCompareSelection()">Cancel</button>';
  document.body.appendChild(bar);
}

function useSuggestion(el) {
  var text = el.querySelector('.suggestion-text').textContent;
  document.getElementById('welcomeInput').value = text;
  sendFromWelcome();
}

// ── Welcome <-> conversation mode ────────────────────────────────────────────

function enterConversationMode() {
  if (conversationStarted) return;
  conversationStarted = true;
  document.getElementById('welcomeScreen').style.display = 'none';
  document.getElementById('messages').classList.add('show');
  document.getElementById('inputBar').classList.add('show');
}

function backToWelcome() {
  conversationStarted = false;
  chatHistory = [];
  document.getElementById('messages').classList.remove('show');
  document.getElementById('messages').innerHTML = '';
  document.getElementById('inputBar').classList.remove('show');
  document.getElementById('welcomeScreen').style.display = 'flex';
  // Full reset, same as what a hard refresh used to be relied on for —
  // clears any sticky compare scope/folder filter and closes any open
  // PDF/text viewer so the welcome screen doesn't come back looking like
  // mid-conversation state leaked through.
  clearActiveFolder();
  if (compareSelection) cancelCompareSelection();
  closePdfPanel();
  closeTxtPanel();
}

// ── Recent conversations (session-local — no backend persistence exists) ────

const RECENT_KEY = 'kb_recent_questions';
function loadRecent() {
  try { return JSON.parse(sessionStorage.getItem(RECENT_KEY) || '[]'); } catch(e) { return []; }
}
function pushRecent(question) {
  var list = loadRecent().filter(function(q) { return q !== question; });
  list.unshift(question);
  list = list.slice(0, 6);
  sessionStorage.setItem(RECENT_KEY, JSON.stringify(list));
  renderRecent();
}
function renderRecent() {
  var list = loadRecent();
  var wrap = document.getElementById('recentWrap');
  // A single entry is almost always just the one suggestion card the user
  // clicked — showing it back as "recent" reads as noise, not a real history.
  if (list.length < 2) { wrap.style.display = 'none'; return; }
  wrap.style.display = 'block';
  document.getElementById('recentList').innerHTML = list.map(function(q) {
    return '<div class="recent-item" role="button" tabindex="0" onclick="rerunRecent(this)">' + svgIcon('message-circle', 14)
      + '<span class="recent-item-text">' + esc(q) + '</span></div>';
  }).join('');
}
function rerunRecent(el) {
  var text = el.querySelector('.recent-item-text').textContent;
  document.getElementById('welcomeInput').value = text;
  sendFromWelcome();
}

function sendFromWelcome() {
  const input = document.getElementById('welcomeInput');
  const text = input.value.trim();
  if (!text || isTyping) return;
  input.value = '';
  enterConversationMode();
  runQuery(text, 3);
}

// ── Messages ──────────────────────────────────────────────────────────────────

function addUserMessage(text) {
  const c = document.getElementById('messages');
  const d = document.createElement('div');
  d.className = 'msg-user';
  d.innerHTML = '<div class="msg-user-bubble">' + esc(text) + '</div>';
  c.appendChild(d); scrollBottom();
}

function showTyping() {
  const c = document.getElementById('messages');
  const d = document.createElement('div');
  d.className = 'typing-row'; d.id = 'typingIndicator';
  d.innerHTML = '<div class="typing-bubble"><span></span><span></span><span></span></div>';
  c.appendChild(d); scrollBottom();
}

function hideTyping() { const t = document.getElementById('typingIndicator'); if (t) t.remove(); }

function addErrorMessage(text) {
  const c = document.getElementById('messages');
  const d = document.createElement('div');
  d.innerHTML = '<div class="error-msg">' + svgIcon('alert-triangle', 15) + '<span>' + esc(text) + '</span></div>';
  c.appendChild(d); scrollBottom();
}

// ── Sources store (avoids inline-onclick escaping bugs) ───────────────────────
var _sourcesStore = [];

function buildSourcesColumn(sources) {
  if (!sources || !sources.length) return '';
  const baseIdx = _sourcesStore.length;
  sources.forEach(function(s) { _sourcesStore.push(s); });

  let html = '<div class="sources-col"><div class="sources-col-label">Sources</div>';
  sources.forEach(function(s, i) {
    const idx = baseIdx + i;
    const doc = Object.values(docsData).find(function(x){ return x.doc_id === s.document; }) || {};
    const fname = doc.filename ? doc.filename.replace(/\.pdf$/i, '') : (s.document || '?');
    html += '<div class="source-card" data-src="' + idx + '" role="button" tabindex="0">';
    html += '<div class="source-num">' + (i + 1) + '</div>';
    html += '<div class="source-body">';
    html += '<div class="source-top"><span class="source-filename" title="' + esc(doc.filename || '') + '">' + esc(fname) + '</span>';
    html += '<span class="source-page">p.' + (s.page || '?') + '</span></div>';
    html += '<div class="source-excerpt">' + esc(s.excerpt || '') + '</div>';
    html += '<div class="source-open">' + svgIcon('external-link', 11) + ' Open document</div>';
    html += '</div></div>';
  });
  html += '</div>';
  return html;
}

function buildAnswerBlock(sources) {
  const hasSources = sources && sources.length > 0;
  const d = document.createElement('div');
  d.className = 'answer-block' + (hasSources ? '' : ' no-sources');
  const answerCard = document.createElement('div');
  answerCard.className = 'answer-card';
  const answerText = document.createElement('div');
  answerText.className = 'answer-text';
  answerCard.appendChild(answerText);
  d.appendChild(answerCard);
  return { wrap: d, answerCard: answerCard, answerText: answerText };
}

function finishAnswerBlock(wrap, answerCard, sources, fullText) {
  const actions = document.createElement('div');
  actions.className = 'answer-actions';
  actions.innerHTML =
    '<button class="answer-action" onclick="copyAnswer(this)">' + svgIcon('file', 12) + ' Copy</button>' +
    '<button class="answer-action" onclick="toggleRetrievalPanel()">' + svgIcon('activity', 12) + ' Retrieval details</button>';
  answerCard.appendChild(actions);
  answerCard.dataset.fullText = fullText;
  if (sources && sources.length) {
    // buildAnswerBlock() had to render single-column before sources were
    // known (they only arrive after the stream's 'sources' event) — flip
    // the wrapper into the two-column layout now that we actually have them.
    wrap.classList.remove('no-sources');
    wrap.insertAdjacentHTML('beforeend', buildSourcesColumn(sources));
  }
}

function copyAnswer(btn) {
  var card = btn.closest('.answer-card');
  var text = card ? card.dataset.fullText : '';
  if (!text) return;
  navigator.clipboard.writeText(text).then(function() {
    var original = btn.innerHTML;
    btn.innerHTML = svgIcon('check', 12) + ' Copied';
    btn.classList.add('copied');
    setTimeout(function() { btn.innerHTML = original; btn.classList.remove('copied'); }, 1600);
  });
}

// ── Send ──────────────────────────────────────────────────────────────────────

function sendMessage(topK) {
  const input = document.getElementById('chatInput');
  const text = input.value.trim();
  if (!text || isTyping) return;
  input.value = '';
  runQuery(text, topK || 3, activeDocumentIds);
}

function runQuery(text, topK, documentIds) {
  if (isTyping) return;
  isTyping = true;
  topK = topK || 3;
  addUserMessage(text); showTyping();
  pushRecent(text);

  // DeepSeek cloud mode has no streaming support in the backend on purpose
  // (rag/generator.py's GeneratorRouter — a request explicitly asking for
  // it on the streaming endpoint gets a clear error rather than a silent
  // switch to local) — so this branch calls the plain /query endpoint and
  // renders the whole answer at once instead of animating token-by-token.
  if (currentProvider === 'deepseek') {
    runQueryNonStreaming(text, topK, documentIds);
    return;
  }

  var accum = '';
  var blockWrap = null;
  var answerCard = null;
  var answerTextEl = null;
  var tokenQueue = [];
  var draining = false;
  var pendingSources = null;
  var streamDone = false;

  function drainQueue() {
    if (!tokenQueue.length) {
      draining = false;
      if (pendingSources) {
        finishAnswerBlock(blockWrap, answerCard, pendingSources.sources, accum);
        scrollBottom();
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
    if (!blockWrap) {
      hideTyping();
      const c = document.getElementById('messages');
      const built = buildAnswerBlock(null);
      blockWrap = built.wrap; answerCard = built.answerCard; answerTextEl = built.answerText;
      c.appendChild(blockWrap);
    }
    accum += token;
    answerTextEl.textContent = accum;
    scrollBottom();
    setTimeout(drainQueue, 18);
  }

  var folderFilter = _folderFilterValue || null;
  apiQueryStream(text, topK, currentModel, chatHistory, folderFilter, documentIds,
    function onToken(token) {
      var clean = token.replace(/[\u3000-\u9fff\uf900-\ufaff\ufe30-\ufe4f\uff00-\uffef]/g, '');
      if (!clean) return;
      tokenQueue.push(clean);
      if (!draining) drainQueue();
    },
    function onSources(sources, debug) {
      pendingSources = { sources: sources, debug: debug };
      if (!draining) drainQueue();
    },
    function onDone(err) {
      if (err) {
        hideTyping();
        if (answerTextEl) {
          var hint = err.partial ? 'Response cut off' : 'Connection error';
          answerTextEl.textContent += '\n\n⚠ ' + hint;
        } else {
          addErrorMessage(err.partial ? 'Response cut off. Please try again.' : 'No connection to the server.');
        }
        isTyping = false;
        return;
      }
      streamDone = true;
      if (!draining) isTyping = false;
    }
  );
}

function runQueryNonStreaming(text, topK, documentIds) {
  var folderFilter = _folderFilterValue || null;
  // currentModel is always an Ollama model name (from the local model
  // picker) — never send it when the cloud provider is selected, or it
  // would override the administrator-configured DeepSeek model server-side.
  var modelForQuery = currentProvider === 'deepseek' ? null : currentModel;
  apiQuery(text, topK, true, modelForQuery, currentProvider, chatHistory, folderFilter, documentIds)
    .then(function(res) {
      hideTyping();
      if (!res.ok || !res.data) {
        addErrorMessage(res.error || 'No connection to the server.');
        isTyping = false;
        return;
      }
      var data = res.data;
      var built = buildAnswerBlock(data.sources);
      document.getElementById('messages').appendChild(built.wrap);
      built.answerText.textContent = data.answer;
      finishAnswerBlock(built.wrap, built.answerCard, data.sources, data.answer);
      scrollBottom();
      updateDebugPanel({ sources: data.sources, debug: Object.assign({}, data.debug, { provider: data.provider, model: data.model }) });
      chatHistory.push({ role: 'user', content: text });
      chatHistory.push({ role: 'assistant', content: data.answer });
      isTyping = false;
    })
    .catch(function() {
      hideTyping();
      addErrorMessage('No connection to the server.');
      isTyping = false;
    });
}

// ── Retrieval details (formerly "debug") ─────────────────────────────────────

function toggleRetrievalPanel() {
  debugVisible = !debugVisible;
  document.getElementById('retrievalPanel').classList.toggle('open', debugVisible);
  document.getElementById('retrievalToggle').classList.toggle('active', debugVisible);
  var sw = document.getElementById('retrievalSwitch');
  if (sw) sw.classList.toggle('on', debugVisible);
}
function updateDebugPanel(data) {
  if (!data.debug) return;
  const d = data.debug;
  document.getElementById('dbgExpansion').textContent = d.expansion_ms || '—';
  document.getElementById('dbgRetrieval').textContent = d.retrieval_ms || '—';
  document.getElementById('dbgRerank').textContent = d.rerank_ms || '—';
  document.getElementById('dbgGeneration').textContent = d.generation_ms || '—';
  document.getElementById('dbgTotal').textContent = d.total_ms || '—';
  document.getElementById('dbgChunks').textContent = (d.chunks_after_rerank || '?') + '/' + (d.chunks_retrieved || '?');
  document.getElementById('dbgScore').textContent = d.best_score != null ? d.best_score.toFixed(3) + ' / avg ' + (d.avg_score || 0).toFixed(3) : '—';
  var providerLabel = d.provider === 'deepseek' ? ' (cloud)' : d.provider === 'local' ? ' (local)' : '';
  document.getElementById('dbgModel').textContent = (d.model || currentModel || '') + providerLabel;
  document.getElementById('dbgQueries').innerHTML = (d.expanded_queries || []).map(function(q, i) {
    return '<div class="dbg-query-item' + (i === 0 ? ' primary' : '') + '">' + esc(q) + '</div>';
  }).join('');
  document.getElementById('dbgChunksList').innerHTML = (d.top_chunks || []).map(function(c) {
    const pct = Math.min(100, c.score * 100).toFixed(0);
    return '<div class="dbg-chunk-item">'
      + '<span class="dbg-chunk-score">' + c.score.toFixed(3) + '</span>'
      + '<div class="dbg-score-bar-wrap"><div class="dbg-score-bar" style="width:' + pct + '%"></div></div>'
      + '<span class="dbg-chunk-source ' + (c.source||'') + '">' + esc(c.source || 'vec') + '</span>'
      + '<span class="dbg-chunk-page">p.' + c.page_num + '</span>'
      + '<span class="dbg-chunk-text">' + esc(c.text_preview) + '</span>'
      + '</div>';
  }).join('');
}

// ── PDF viewer ────────────────────────────────────────────────────────────────

if (typeof pdfjsLib !== 'undefined') {
  pdfjsLib.GlobalWorkerOptions.workerSrc = 'vendor/pdfjs/pdf.worker.min.js';
}
let pdfDoc = null, currentPage = 1, currentDocId = null, isRendering = false;
// pendingPage previously stored only a bare page number — opening a
// *different* document at the same page number it was already showing (a
// coincidence, not an edge case: "page 1" is the common case) made
// `next !== pageNum` false, so the queued render was silently dropped
// instead of ever firing. renderGen is bumped on every render *request*
// (open or page-change); any async step checks it against the current
// value and bails if a newer request has since superseded it — this is
// what actually distinguishes "same page, different document" rather than
// page number ever could.
let pendingRender = null; // { docId, pageNum, gen } queued while a render is in flight
let renderGen = 0;

function setPdfNavDisabled(disabled) {
  const prev = document.getElementById('pdfPrevBtn');
  const next = document.getElementById('pdfNextBtn');
  if (prev) prev.disabled = disabled;
  if (next) next.disabled = disabled;
}

async function openPdfViewer(docId, page) {
  const doc = docsData[docId] || {};
  document.getElementById('pdfPanelTitle').textContent = doc.filename || docId;
  document.getElementById('pdfPanel').classList.add('open');
  document.getElementById('pdfOverlay').classList.add('show');
  const myGen = ++renderGen; // every open/page-change request gets its own token
  if (currentDocId !== docId) {
    currentDocId = docId; pdfDoc = null; pendingRender = null;
    document.getElementById('pdfPageWrapper').style.display = 'none';
    document.getElementById('pdfLoading').style.display = 'flex';
    document.getElementById('pdfLoading').innerHTML = svgIcon('file-text', 26) + '<div>Loading…</div>';
    try {
      const loaded = await pdfjsLib.getDocument(getPdfLoadOptions(docId)).promise;
      if (myGen !== renderGen) return; // a newer open/page-change fired while this PDF was loading
      pdfDoc = loaded;
      document.getElementById('pdfLoading').style.display = 'none';
      document.getElementById('pdfPageWrapper').style.display = 'inline-block';
    } catch(e) {
      if (myGen !== renderGen) return;
      document.getElementById('pdfLoading').innerHTML = svgIcon('alert-triangle', 26) + '<div>Failed to load PDF</div>';
      return;
    }
  }
  currentPage = page || 1;
  await renderPage(currentPage, docId, myGen);
}

async function renderPage(pageNum, docId, gen) {
  if (!pdfDoc || gen !== renderGen) return; // superseded before we even started
  if (isRendering) {
    // Don't drop the request — a fast double-click on "next" (or opening a
    // different source while a render is in flight) used to just vanish
    // silently. Remember the latest ask (which document, which page, which
    // generation) and pick it up once the in-flight render finishes.
    pendingRender = { docId, pageNum, gen };
    return;
  }
  isRendering = true;
  setPdfNavDisabled(true);
  try {
    const page = await pdfDoc.getPage(pageNum);
    if (gen !== renderGen) return; // superseded mid-fetch — don't touch the canvas for a stale request
    const canvas = document.getElementById('pdfCanvas');
    const wrapper = document.getElementById('pdfPageWrapper');
    const container = document.getElementById('pdfCanvasContainer');
    const scale = (container.clientWidth - 28) / page.getViewport({ scale: 1 }).width;
    const vp = page.getViewport({ scale });

    canvas.width = vp.width;
    canvas.height = vp.height;
    wrapper.style.width = vp.width + 'px';
    wrapper.style.height = vp.height + 'px';

    await page.render({ canvasContext: canvas.getContext('2d'), viewport: vp }).promise;
    if (gen !== renderGen) return; // superseded while the canvas was painting

    document.getElementById('pdfPageInfo').textContent = 'Page ' + pageNum + ' of ' + pdfDoc.numPages;
    wrapper.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (e) {
    console.warn('PDF render failed:', e);
  } finally {
    // Always clears, even on a PDF.js render error — the old code only set
    // this on the success path, so one failed render permanently wedged
    // page navigation for the rest of the session (isRendering stuck true).
    isRendering = false;
    setPdfNavDisabled(false);
    // Must live *inside* finally, not after the try/finally statement — a
    // `return` in the try block (the "superseded" bail-outs above) runs
    // finally and then exits the function immediately; code placed after
    // the whole try/finally never executes on that path, so a render that
    // got superseded never drained its own pendingRender. It looked fine
    // whenever the try body ran to completion normally and only broke on
    // exactly the case this exists to handle.
    const next = pendingRender;
    pendingRender = null;
    if (next && next.gen === renderGen) {
      void renderPage(next.pageNum, next.docId, next.gen);
    }
  }
}

async function changePage(delta) {
  if (!pdfDoc) return;
  const np = currentPage + delta;
  if (np < 1 || np > pdfDoc.numPages) return;
  currentPage = np;
  const myGen = ++renderGen;
  await renderPage(currentPage, currentDocId, myGen);
}

function closePdfPanel() {
  document.getElementById('pdfPanel').classList.remove('open');
  document.getElementById('pdfOverlay').classList.remove('show');
}

// ── TXT viewer ────────────────────────────────────────────────────────────────
// Canonical source view for BOTH txt and pdf documents — offset-based
// highlighting: char_start/char_end come straight from the query response,
// which are indices into the exact same normalize_whitespace() text the
// backend hands back (persisted at ingestion for PDF, read live off disk
// for TXT — see get_document_page() in api/main.py), so no fuzzy search
// against a rendered page is ever needed. PDF's own layout (images, stamps,
// QR codes, original pagination) is a click away via "View original PDF",
// which opens the real PDF.js view with no highlight promised there.

let currentTxtDocId = null, currentTxtPage = 1, currentTxtTotalPages = 1;
// Bumped on every openTextViewer() call — an in-flight fetch for a
// previously-opened document/page checks its own token against the current
// value before touching the DOM, so a slow response for document A opened
// just before document B can't paint A's text under B's title (the PDF
// viewer's renderGen plays the identical role for openPdfViewer()).
let txtViewGen = 0;

async function openTextViewer(docId, page, charStart, charEnd) {
  const doc = docsData[docId] || {};
  const myGen = ++txtViewGen;
  document.getElementById('txtPanelTitle').textContent = doc.filename || docId;
  document.getElementById('txtPanel').classList.add('open');
  document.getElementById('pdfOverlay').classList.add('show');
  const body = document.getElementById('txtPanelBody');
  body.textContent = 'Loading…';
  document.getElementById('txtPageControls').style.display = 'none';
  currentTxtDocId = docId;
  currentTxtPage = page || 1;
  try {
    const data = await apiGetDocumentPage(docId, currentTxtPage);
    if (myGen !== txtViewGen) return; // superseded by a newer open/page-change while this was in flight
    currentTxtPage = data.page;
    currentTxtTotalPages = data.total_pages;
    const text = data.text || '';
    let html;
    if (charStart != null && charEnd != null && charStart >= 0 && charEnd <= text.length && charStart < charEnd) {
      html = esc(text.slice(0, charStart)) +
             '<mark id="txtHighlight">' + esc(text.slice(charStart, charEnd)) + '</mark>' +
             esc(text.slice(charEnd));
    } else {
      html = esc(text);
    }
    body.innerHTML = html;
    const mark = document.getElementById('txtHighlight');
    if (mark) mark.scrollIntoView({ behavior: 'smooth', block: 'center' });

    if (data.format === 'pdf') {
      document.getElementById('txtPageControls').style.display = 'flex';
      document.getElementById('txtPageInfo').textContent = 'Page ' + currentTxtPage + ' of ' + currentTxtTotalPages;
      document.getElementById('txtPrevBtn').disabled = currentTxtPage <= 1;
      document.getElementById('txtNextBtn').disabled = currentTxtPage >= currentTxtTotalPages;
    }
  } catch (e) {
    if (myGen !== txtViewGen) return;
    body.textContent = 'Failed to load document.';
  }
}

async function changeTxtPage(delta) {
  if (!currentTxtDocId) return;
  const np = currentTxtPage + delta;
  if (np < 1 || np > currentTxtTotalPages) return;
  await openTextViewer(currentTxtDocId, np, null, null);
}

function viewOriginalPdf() {
  if (!currentTxtDocId) return;
  // Both panels are `position: fixed; right: 0` at the same z-index when
  // open — txt-panel sits later in the DOM, so leaving it open while
  // openPdfViewer() renders underneath it made the button look like it did
  // nothing (the PDF loaded, just invisible behind the still-open txt
  // panel).
  closeTxtPanel();
  openPdfViewer(currentTxtDocId, currentTxtPage);
}

// Reverse of viewOriginalPdf() above — openPdfViewer() only ever runs for a
// PDF-format document (its only caller is viewOriginalPdf(), itself only
// reachable once openTextViewer() has already confirmed format === 'pdf'),
// so get_document_page() always has a page to show here; no format check
// needed on this side. currentDocId/currentPage are the PDF panel's own
// state (see openPdfViewer()/changePage()), same relationship as
// currentTxtDocId/currentTxtPage above.
function viewOriginalText() {
  if (!currentDocId) return;
  closePdfPanel();
  openTextViewer(currentDocId, currentPage, null, null);
}

function closeTxtPanel() {
  document.getElementById('txtPanel').classList.remove('open');
  document.getElementById('pdfOverlay').classList.remove('show');
}

// ── API Key ───────────────────────────────────────────────────────────────────

function openKeyModal() {
  document.getElementById('settingsPanel').classList.remove('show');
  document.getElementById('keyInput').value = localStorage.getItem('api_key') || '';
  document.getElementById('keyOverlay').classList.add('show');
  document.getElementById('keyInput').focus();
}

function closeKeyModal() {
  document.getElementById('keyOverlay').classList.remove('show');
}

function saveApiKey() {
  const val = document.getElementById('keyInput').value.trim();
  localStorage.setItem('api_key', val);
  closeKeyModal();
  checkHealth();
  loadDocuments();
  loadModels();
}

// ── Init ──────────────────────────────────────────────────────────────────────

injectStaticIcons();
renderRecent();
if (!localStorage.getItem('api_key')) {
  openKeyModal();
}
checkHealth();
loadDocuments();
loadModels();
setInterval(checkHealth, 15000);
setInterval(loadModels, 60000);
