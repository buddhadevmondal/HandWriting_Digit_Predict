(() => {
  'use strict';

  const API_URL = '/predict';
  const MAX_BYTES = 8 * 1024 * 1024;
  const ALLOWED_TYPES = [
    'image/png',
    'image/jpeg',
    'image/webp',
    'image/bmp'
  ];
  const ALLOWED_EXT = ['png', 'jpg', 'jpeg', 'webp', 'bmp'];

  const $ = (id) => document.getElementById(id);

  const el = {
    drop: $('dropzone'),
    empty: $('dropEmpty'),
    preview: $('preview'),
    img: $('previewImg'),
    input: $('fileInput'),
    browse: $('browseBtn'),
    change: $('changeBtn'),
    remove: $('removeBtn'),
    meta: $('fileMeta'),
    error: $('error'),
    predict: $('predictBtn'),
    label: $('predictLabel'),
    idle: $('stateIdle'),
    loading: $('stateLoading'),
    done: $('stateDone'),
    digit: $('bigDigit'),
    conf: $('confNum'),
    bars: $('bars'),
    note: $('note'),
    reset: $('resetBtn')
  };

  let file = null;
  let previewUrl = null;
  let busy = false;
  let controller = null;

  function showState(name) {
    el.idle.hidden = name !== 'idle';
    el.loading.hidden = name !== 'loading';
    el.done.hidden = name !== 'done';
  }

  function setError(message) {
    el.error.textContent = message || '';
    el.error.hidden = !message;
  }

  function formatSize(bytes) {
    return bytes >= 1048576
      ? (bytes / 1048576).toFixed(2) + ' MB'
      : Math.max(1, Math.round(bytes / 1024)) + ' KB';
  }

  function validate(f) {
    const ext = (f.name.split('.').pop() || '').toLowerCase();

    if (!ALLOWED_TYPES.includes(f.type) && !ALLOWED_EXT.includes(ext)) {
      return 'Unsupported file type. Choose PNG, JPG, JPEG, WEBP or BMP.';
    }

    if (f.size > MAX_BYTES) {
      return 'Image must be 8 MB or smaller.';
    }

    if (f.size === 0) {
      return 'The selected file is empty.';
    }

    return null;
  }

  function setFile(f) {
    const problem = validate(f);

    if (problem) {
      setError(problem);
      return;
    }

    setError('');

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

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

    showState('idle');
  }

  function clearFile() {
    if (controller) {
      controller.abort();
    }

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    file = null;
    previewUrl = null;

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
    el.label.textContent = on ? 'Recognizing...' : 'Recognize Digit';

    el.change.disabled = on;
    el.remove.disabled = on;
  }

  function toPercent(value) {
    const n = Number(value);

    if (!Number.isFinite(n)) {
      return null;
    }

    return n <= 1 ? n * 100 : n;
  }

  function pctText(percent) {
    return (percent >= 99.95 ? '100' : percent.toFixed(1)) + '%';
  }

  function renderResult(data) {
    // Supports the response format used by main.py.
    const digit = data.prediction ?? data.predicted_digit;
    const confidence = toPercent(data.confidence);

    if (
      digit === undefined ||
      digit === null ||
      confidence === null
    ) {
      throw new Error(
        'Prediction is missing from the server response. Check main.py.'
      );
    }

    el.digit.textContent = String(digit);
    el.conf.textContent = pctText(confidence);

    el.bars.replaceChildren();

    const predictions = Array.isArray(data.top_predictions)
      ? data.top_predictions.slice(0, 3)
      : [];

    predictions.forEach((item) => {
      const d =
        item.digit ??
        item.label ??
        item.class ??
        item.predicted_digit;

      const p = toPercent(
        item.confidence ?? item.probability ?? item.score
      );

      if (d === undefined || p === null) {
        return;
      }

      const li = document.createElement('li');
      li.className = 'bar-row';

      li.innerHTML =
        '<span class="bd"></span>' +
        '<span class="track"><span class="fill"></span></span>' +
        '<span class="pct"></span>';

      li.querySelector('.bd').textContent = d;
      li.querySelector('.pct').textContent = pctText(p);

      li.querySelector('.track').setAttribute('role', 'img');
      li.querySelector('.track').setAttribute(
        'aria-label',
        `Digit ${d}: ${pctText(p)}`
      );

      el.bars.appendChild(li);

      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          li.querySelector('.fill').style.width =
            Math.min(100, Math.max(0, p)) + '%';
        });
      });
    });

    el.note.hidden = !data.note;
    el.note.textContent = data.note || '';

    showState('done');
  }

  async function predict() {
    if (!file || busy) {
      return;
    }

    setError('');
    setBusy(true);
    showState('loading');

    controller = new AbortController();

    try {
      const body = new FormData();
      body.append('file', file);

      const response = await fetch(API_URL, {
        method: 'POST',
        body,
        signal: controller.signal
      });

      let data = null;

      try {
        data = await response.json();
      } catch {
        // Handle non-JSON server responses below.
      }

      if (!response.ok) {
        let detail = data && data.detail;

        if (Array.isArray(detail)) {
          detail = detail
            .map((item) => item.msg || JSON.stringify(item))
            .join(' ');
        } else if (detail && typeof detail === 'object') {
          detail = JSON.stringify(detail);
        }

        throw new Error(
          detail || `Server error (${response.status}).`
        );
      }

      if (!data) {
        throw new Error('The server returned an unreadable response.');
      }

      renderResult(data);

    } catch (error) {
      if (error.name === 'AbortError') {
        return;
      }

      showState('idle');

      setError(
        error instanceof TypeError
          ? 'Cannot connect to the server. Check that FastAPI is running.'
          : error.message
      );

    } finally {
      controller = null;
      setBusy(false);
    }
  }

  // File selection
  el.browse.addEventListener('click', (event) => {
    event.stopPropagation();
    el.input.click();
  });

  el.change.addEventListener('click', () => {
    el.input.click();
  });

  el.input.addEventListener('change', () => {
    if (el.input.files[0]) {
      setFile(el.input.files[0]);
    }

    el.input.value = '';
  });

  // Remove and reset
  el.remove.addEventListener('click', () => {
    clearFile();
    setError('');
    showState('idle');
  });

  el.reset.addEventListener('click', resetAll);

  // Prediction button
  el.predict.addEventListener('click', predict);

  // Dropzone interaction
  el.drop.addEventListener('click', () => {
    if (!file) {
      el.input.click();
    }
  });

  el.drop.addEventListener('keydown', (event) => {
    if (
      !file &&
      (event.key === 'Enter' || event.key === ' ')
    ) {
      event.preventDefault();
      el.input.click();
    }
  });

  ['dragenter', 'dragover'].forEach((eventName) => {
    el.drop.addEventListener(eventName, (event) => {
      event.preventDefault();

      if (!busy) {
        el.drop.classList.add('over');
      }
    });
  });

  ['dragleave', 'drop'].forEach((eventName) => {
    el.drop.addEventListener(eventName, (event) => {
      event.preventDefault();
      el.drop.classList.remove('over');
    });
  });

  el.drop.addEventListener('drop', (event) => {
    if (busy) {
      return;
    }

    const droppedFile =
      event.dataTransfer.files &&
      event.dataTransfer.files[0];

    if (droppedFile) {
      setFile(droppedFile);
    }
  });

  // Prevent images opening when dropped outside the dropzone.
  ['dragover', 'drop'].forEach((eventName) => {
    window.addEventListener(eventName, (event) => {
      event.preventDefault();
    });
  });
})();