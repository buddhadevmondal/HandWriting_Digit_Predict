(() => {
  'use strict';

  // The FastAPI endpoint. Change the port here if your server runs elsewhere.
  const API_URL = 'http://127.0.0.1:8000/api/predict';
  const MAX_BYTES = 8 * 1024 * 1024;
  const ALLOWED_TYPES = ['image/png', 'image/jpeg', 'image/webp', 'image/bmp'];
  const ALLOWED_EXT = ['png', 'jpg', 'jpeg', 'webp', 'bmp'];

  const $ = (id) => document.getElementById(id);
  
  const el = {
    drop: $('dropzone'), empty: $('dropEmpty'), preview: $('preview'), img: $('previewImg'),
    input: $('fileInput'), browse: $('browseBtn'), change: $('changeBtn'), remove: $('removeBtn'),
    meta: $('fileMeta'), error: $('error'), predict: $('predictBtn'), label: $('predictLabel'),
    idle: $('stateIdle'), loading: $('stateLoading'), done: $('stateDone'),
    digit: $('bigDigit'), conf: $('confNum'), bars: $('bars'), note: $('note'), reset: $('resetBtn'),
  };

  let file = null;
  let previewUrl = null;
  let busy = false;
  let controller = null;

  const showState = (name) => {
    el.idle.hidden = name !== 'idle';
    el.loading.hidden = name !== 'loading';
    el.done.hidden = name !== 'done';
  };
  const setError = (msg) => { el.error.textContent = msg || ''; el.error.hidden = !msg; };
  const formatSize = (b) => (b >= 1048576 ? (b / 1048576).toFixed(2) + ' MB' : Math.max(1, Math.round(b / 1024)) + ' KB');

  function validate(f) {
    const ext = (f.name.split('.').pop() || '').toLowerCase();
    if (!ALLOWED_TYPES.includes(f.type) && !ALLOWED_EXT.includes(ext)) {
      return 'That file type is not supported. Please choose a PNG, JPG, JPEG, WEBP or BMP image.';
    }
    if (f.size > MAX_BYTES) return `That image is ${formatSize(f.size)}. Please choose one that is 8 MB or smaller.`;
    if (f.size === 0) return 'That file is empty. Please choose a different image.';
    return null;
  }

  function setFile(f) {
    const problem = validate(f);
    if (problem) { setError(problem); return; }
    setError('');
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    file = f;
    previewUrl = URL.createObjectURL(f);
    el.img.src = previewUrl;
    el.preview.hidden = false;
    el.empty.hidden = true;
    el.drop.classList.add('has');
    el.drop.removeAttribute('tabindex');
    el.drop.removeAttribute('role');
    el.meta.textContent = `${f.name} · ${formatSize(f.size)}`;
    el.meta.hidden = false;
    el.change.hidden = false;
    el.remove.hidden = false;
    el.predict.disabled = false;
    showState('idle'); // clear any earlier result for the old image
  }

  function clearFile() {
    if (controller) controller.abort();
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    file = null; previewUrl = null;
    el.input.value = '';
    el.img.removeAttribute('src');
    el.preview.hidden = true;
    el.empty.hidden = false;
    el.drop.classList.remove('has', 'over');
    el.drop.setAttribute('tabindex', '0');
    el.drop.setAttribute('role', 'button');
    el.meta.hidden = true;
    el.change.hidden = true;
    el.remove.hidden = true;
    el.predict.disabled = true;
  }

  function resetAll() {
    clearFile();
    setError('');
    setBusy(false);
    showState('idle');
    el.drop.focus();
  }

  function setBusy(on) {
    busy = on;
    el.predict.disabled = on || !file;
    el.predict.classList.toggle('busy', on);
    el.predict.setAttribute('aria-busy', String(on));
    el.label.textContent = on ? 'Recognizing…' : 'Recognize Digit';
    el.change.disabled = on;
    el.remove.disabled = on;
  }

  const toPercent = (v) => {
    const n = Number(v);
    if (!Number.isFinite(n)) return null;
    return n <= 1 ? n * 100 : n; // accept 0–1 or 0–100
  };
  const pctText = (p) => (p >= 99.95 ? '100' : p.toFixed(1)) + '%';

  function renderResult(data) {
    const digit = data.predicted_digit;
    const conf = toPercent(data.confidence);
    if (digit === undefined || digit === null || conf === null) {
      throw new Error('The server replied, but the prediction was missing from the response.');
    }
    el.digit.textContent = String(digit);
    el.conf.textContent = pctText(conf);

    el.bars.replaceChildren();
    const top = Array.isArray(data.top_predictions) ? data.top_predictions.slice(0, 3) : [];
    top.forEach((item) => {
      const d = item.digit ?? item.label ?? item.class ?? item.predicted_digit;
      const p = toPercent(item.confidence ?? item.probability ?? item.score);
      if (d === undefined || p === null) return;
      const li = document.createElement('li');
      li.className = 'bar-row';
      li.innerHTML = '<span class="bd"></span><span class="track"><span class="fill"></span></span><span class="pct"></span>';
      li.querySelector('.bd').textContent = d;
      li.querySelector('.pct').textContent = pctText(p);
      li.querySelector('.track').setAttribute('role', 'img');
      li.querySelector('.track').setAttribute('aria-label', `Digit ${d}: ${pctText(p)}`);
      el.bars.appendChild(li);
      requestAnimationFrame(() => requestAnimationFrame(() => {
        li.querySelector('.fill').style.width = Math.min(100, Math.max(0, p)) + '%';
      }));
    });

    el.note.hidden = !data.note;
    el.note.textContent = data.note || '';
    showState('done');
  }

  async function predict() {
    if (!file || busy) return;
    setError('');
    setBusy(true);
    showState('loading');
    controller = new AbortController();

    try {
      const body = new FormData();
      body.append('file', file);
      const res = await fetch(API_URL, { method: 'POST', body, signal: controller.signal });

      let data = null;
      try { data = await res.json(); } catch { /* non-JSON response */ }

      if (!res.ok) {
        let detail = data && data.detail;
        if (Array.isArray(detail)) detail = detail.map((d) => d.msg || JSON.stringify(d)).join(' ');
        else if (detail && typeof detail === 'object') detail = JSON.stringify(detail);
        throw new Error(detail || `The server returned an error (${res.status}). Please try again.`);
      }
      if (!data) throw new Error('The server sent a response that could not be read.');
      renderResult(data);
    } catch (err) {
      if (err.name === 'AbortError') return;
      showState('idle');
      setError(err instanceof TypeError
        ? 'Could not reach the recognition server. Check that the FastAPI app is running and that CORS allows this page.'
        : err.message);
    } finally {
      controller = null;
      setBusy(false);
    }
  }

  // events
  el.browse.addEventListener('click', (e) => { e.stopPropagation(); el.input.click(); });
  el.change.addEventListener('click', () => el.input.click());
  el.remove.addEventListener('click', () => { clearFile(); setError(''); showState('idle'); });
  el.reset.addEventListener('click', resetAll);
  el.predict.addEventListener('click', predict);
  el.input.addEventListener('change', () => { if (el.input.files[0]) setFile(el.input.files[0]); el.input.value = ''; });

  el.drop.addEventListener('click', () => { if (!file) el.input.click(); });
  el.drop.addEventListener('keydown', (e) => {
    if (!file && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); el.input.click(); }
  });
  ['dragenter', 'dragover'].forEach((t) => el.drop.addEventListener(t, (e) => {
    e.preventDefault(); if (!busy) el.drop.classList.add('over');
  }));
  ['dragleave', 'drop'].forEach((t) => el.drop.addEventListener(t, (e) => {
    e.preventDefault(); el.drop.classList.remove('over');
  }));
  el.drop.addEventListener('drop', (e) => {
    if (busy) return;
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) setFile(f);
  });
  // stop the browser from opening images dropped outside the zone
  ['dragover', 'drop'].forEach((t) => window.addEventListener(t, (e) => e.preventDefault()));
})();
