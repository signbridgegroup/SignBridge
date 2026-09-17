/**
 * Motion-library browser: lists every token across all three recognition
 * datasets from the existing GET /api/motions endpoint (one lightweight
 * metadata fetch, ~140 KB for 1,347 entries — no motion JSON is fetched
 * here), with client-side dataset/status filtering and Arabic/English
 * token search. Pressing Play on a row is the only thing that ever fetches
 * an actual motion JSON, via the caller-supplied onPlay callback, which
 * reuses the existing MotionQueuePlayer exactly like Path A/B already do.
 */
import { fetchMotionsList } from './api.js';
import { t, onLanguageChange } from './i18n.js';

const DATASET_LABEL_KEYS = {
  jordanian_it: 'library.dataset.jordanian_it',
  karsl: 'library.dataset.karsl',
  isharah: 'library.dataset.isharah',
};

const STATUS_LABEL_KEYS = {
  ready: 'library.status_label.ready',
  pending_generation: 'library.status_label.pending_generation',
  source_unavailable: 'library.status_label.source_unavailable',
  incompatible_source: 'library.status_label.incompatible_source',
  extraction_failed: 'library.status_label.extraction_failed',
  invalid_motion: 'library.status_label.invalid_motion',
  validation_required: 'library.status_label.validation_required',
};

const KNOWN_UNAVAILABLE_ISHARAH_GLOSSES = ['اليوم', 'بحرهو', 'رجوع', 'سيجاره', 'طاوله', 'وفاه'];

// Cap how many rows are actually rendered at once so 1,347 entries stay
// responsive without needing a virtualization library. Narrowing the
// search/filter naturally brings the visible set below this cap.
const MAX_RENDERED_ROWS = 250;

export class MotionLibraryBrowser {
  constructor(root, { onPlay } = {}) {
    this.root = root;
    this.onPlay = onPlay;
    this.allEntries = [];
    this.loaded = false;
    this.loading = false;
    this.datasetFilter = 'all';
    this.statusFilter = 'all';
    this.searchQuery = '';
    this.playingToken = null;

    this._buildShell();
    onLanguageChange(() => {
      this._applyStaticText();
      this._render();
    });
  }

  _buildShell() {
    this.root.innerHTML = `
      <div class="motion-library">
        <div class="motion-library-controls">
          <input type="search" class="motion-library-search" />
          <div class="motion-library-filters" role="group" data-i18n-aria-label="library.filter.dataset">
            <button type="button" class="filter-chip is-active" data-dataset="all"></button>
            <button type="button" class="filter-chip" data-dataset="jordanian_it"></button>
            <button type="button" class="filter-chip" data-dataset="karsl"></button>
            <button type="button" class="filter-chip" data-dataset="isharah"></button>
          </div>
          <div class="motion-library-filters" role="group" data-i18n-aria-label="library.filter.status">
            <button type="button" class="filter-chip is-active" data-status="all"></button>
            <button type="button" class="filter-chip" data-status="ready"></button>
            <button type="button" class="filter-chip" data-status="source_unavailable"></button>
          </div>
        </div>
        <p class="motion-library-summary" aria-live="polite"></p>
        <div class="motion-library-unavailable-note" hidden>
          <strong class="motion-library-unavailable-title"></strong>
          <span>${KNOWN_UNAVAILABLE_ISHARAH_GLOSSES.join('، ')}</span>
        </div>
        <ul class="motion-library-list"></ul>
      </div>
    `;

    this.searchInput = this.root.querySelector('.motion-library-search');
    this.summaryEl = this.root.querySelector('.motion-library-summary');
    this.listEl = this.root.querySelector('.motion-library-list');
    this.unavailableNoteEl = this.root.querySelector('.motion-library-unavailable-note');
    this.unavailableTitleEl = this.root.querySelector('.motion-library-unavailable-title');
    this.datasetChips = [...this.root.querySelectorAll('[data-dataset]')];
    this.statusChips = [...this.root.querySelectorAll('[data-status]')];

    this._applyStaticText();

    this.searchInput.addEventListener('input', () => {
      this.searchQuery = this.searchInput.value.trim();
      this._render();
    });
    this.datasetChips.forEach((chip) => {
      chip.addEventListener('click', () => {
        this.datasetFilter = chip.dataset.dataset;
        this.datasetChips.forEach((c) => c.classList.toggle('is-active', c === chip));
        this._render();
      });
    });
    this.statusChips.forEach((chip) => {
      chip.addEventListener('click', () => {
        this.statusFilter = chip.dataset.status;
        this.statusChips.forEach((c) => c.classList.toggle('is-active', c === chip));
        this._render();
      });
    });
  }

  /** Refreshes every static (non-data-driven) string in the shell for the
   * current language — called once at construction and again whenever the
   * interface language changes, without rebuilding the shell (so search
   * text and active filters survive a language switch). */
  _applyStaticText() {
    this.searchInput.setAttribute('placeholder', t('library.search_placeholder'));
    this.searchInput.setAttribute('aria-label', t('library.search_placeholder'));
    this.listEl.setAttribute('aria-label', t('library.title'));
    this.unavailableTitleEl.textContent = t('library.unavailable_note_title');

    const datasetLabels = { all: 'library.filter.all', jordanian_it: 'library.filter.jordanian_it', karsl: 'library.filter.karsl', isharah: 'library.filter.isharah' };
    this.datasetChips.forEach((chip) => {
      chip.textContent = t(datasetLabels[chip.dataset.dataset]);
    });
    const statusLabels = { all: 'library.status.all', ready: 'library.status.ready', source_unavailable: 'library.status.source_unavailable' };
    this.statusChips.forEach((chip) => {
      chip.textContent = t(statusLabels[chip.dataset.status]);
    });
  }

  /** Fetches the manifest listing once (lazily — call when the panel is first opened). */
  async ensureLoaded() {
    if (this.loaded || this.loading) return;
    this.loading = true;
    this.summaryEl.textContent = t('library.loading');
    try {
      const body = await fetchMotionsList();
      this.allEntries = body.tokens || [];
      this.loaded = true;
      this._render();
    } catch (error) {
      this.summaryEl.textContent = t('library.load_error');
      console.error('[MotionLibrary] failed to load /api/motions', error);
    } finally {
      this.loading = false;
    }
  }

  _matches(entry, query) {
    if (!query) return true;
    const needle = query.toLowerCase();
    const token = String(entry.token || '').toLowerCase();
    const label = String(entry.label || '').toLowerCase();
    // Arabic has no case, so .toLowerCase() is a harmless no-op there;
    // this still lets one search box cover both scripts.
    return token.includes(needle) || label.includes(needle);
  }

  _filteredEntries() {
    return this.allEntries.filter((entry) => {
      if (this.datasetFilter !== 'all' && entry.dataset !== this.datasetFilter) return false;
      if (this.statusFilter === 'ready' && entry.status !== 'ready') return false;
      if (this.statusFilter === 'source_unavailable' && entry.status !== 'source_unavailable') return false;
      return this._matches(entry, this.searchQuery);
    });
  }

  _render() {
    if (!this.loaded) return;
    const filtered = this._filteredEntries();

    this.unavailableNoteEl.hidden = !(
      this.datasetFilter === 'isharah' || this.statusFilter === 'source_unavailable'
    );

    const total = this.allEntries.length;
    const readyTotal = this.allEntries.filter((e) => e.status === 'ready').length;
    if (filtered.length > MAX_RENDERED_ROWS) {
      this.summaryEl.textContent = t('library.summary_capped', {
        shown: MAX_RENDERED_ROWS,
        matched: filtered.length,
        total,
        ready: readyTotal,
      });
    } else {
      this.summaryEl.textContent = t('library.summary', { matched: filtered.length, total, ready: readyTotal });
    }

    const rows = filtered.slice(0, MAX_RENDERED_ROWS);
    this.listEl.innerHTML = '';
    if (!rows.length) {
      const empty = document.createElement('li');
      empty.className = 'motion-library-empty';
      empty.textContent = t('library.empty');
      this.listEl.append(empty);
      return;
    }

    const fragment = document.createDocumentFragment();
    for (const entry of rows) {
      fragment.append(this._buildRow(entry));
    }
    this.listEl.append(fragment);
  }

  _buildRow(entry) {
    const li = document.createElement('li');
    li.className = 'motion-library-row';

    const info = document.createElement('div');
    info.className = 'motion-library-row-info';
    const title = document.createElement('span');
    title.className = 'motion-library-row-title';
    title.textContent = entry.label || entry.token;
    const meta = document.createElement('span');
    meta.className = 'motion-library-row-meta';
    const datasetLabel = DATASET_LABEL_KEYS[entry.dataset] ? t(DATASET_LABEL_KEYS[entry.dataset]) : entry.dataset;
    meta.textContent = `${datasetLabel} · ${entry.token}`;
    info.append(title, meta);

    const badge = document.createElement('span');
    const isReady = entry.status === 'ready';
    badge.className = `status-badge ${isReady ? 'status-ready' : 'status-unavailable'}`;
    badge.textContent = STATUS_LABEL_KEYS[entry.status] ? t(STATUS_LABEL_KEYS[entry.status]) : entry.status;

    li.append(info, badge);

    if (isReady) {
      const playButton = document.createElement('button');
      playButton.type = 'button';
      playButton.className = 'secondary-button motion-library-play';
      const isPlaying = this.playingToken === `${entry.dataset}:${entry.token}`;
      playButton.textContent = isPlaying ? t('library.playing') : t('library.play');
      playButton.disabled = isPlaying;
      playButton.addEventListener('click', () => this._handlePlay(entry, playButton));
      li.append(playButton);
    }

    return li;
  }

  async _handlePlay(entry, button) {
    if (!this.onPlay) return;
    const key = `${entry.dataset}:${entry.token}`;
    this.playingToken = key;
    button.disabled = true;
    button.textContent = t('library.playing');
    try {
      // Relative URL, matching the shape the backend already returns in
      // resolved_motion_sequence / recognized_motion_sequence entries — the
      // caller (main.js's playMotionSequence) applies motionUrl() itself,
      // exactly like Path A/B, so the prefix is only ever added once.
      await this.onPlay({
        token: entry.token,
        dataset: entry.dataset,
        status: 'ready',
        motion_url: `/api/motions/${entry.dataset}/${encodeURIComponent(entry.token)}`,
      });
    } finally {
      this.playingToken = null;
      button.disabled = false;
      button.textContent = t('library.play');
    }
  }
}
