/**
 * Sequential avatar motion playback built on top of the unmodified,
 * frozen retargeting module (motion-retarget.js). This file adds only the
 * queueing/UI-notification layer described in the integration brief —
 * it never touches AXIS_SIGN, the palm basis, the IK solver, or any other
 * retargeting math.
 */
import { AvatarMotionRetargeter, BakedAvatarMotion, SignMotionClip } from './motion-retarget.js';

export class MotionQueuePlayer {
  constructor(avatar, { easeInSeconds = 0.4, easeOutSeconds = 0.4 } = {}) {
    this.retargeter = new AvatarMotionRetargeter(avatar);
    this.easeInSeconds = easeInSeconds;
    this.easeOutSeconds = easeOutSeconds;

    this.queue = [];
    this.index = -1;
    this.clipStartedAt = null;
    this.activeMotion = null;
    this.playing = false;

    this.onTokenStart = null;
    this.onTokenEnd = null;
    this.onQueueComplete = null;
  }

  get missingTokens() {
    return this.queue.filter((item) => item.status !== 'ready').map((item) => item.token);
  }

  get currentToken() {
    return this.index >= 0 && this.index < this.queue.length ? this.queue[this.index] : null;
  }

  async loadSequence(entries) {
    this.stop();
    this.queue = [];

    for (const entry of entries) {
      if (entry.status !== 'ready' || !entry.motion_url) {
        this.queue.push({ ...entry, clip: null });
        continue;
      }
      try {
        const response = await fetch(entry.motion_url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        this.queue.push({ ...entry, clip: new SignMotionClip(data) });
      } catch (error) {
        console.error(`[MotionQueue] Failed to load motion for ${entry.token}:`, error);
        this.queue.push({ ...entry, status: 'missing', clip: null });
      }
    }
  }

  play(now) {
    if (!this.queue.length) return;
    this.index = -1;
    this.playing = true;
    this._advance(now);
  }

  replay(now) {
    this.play(now);
  }

  stop() {
    this.playing = false;
    this.index = -1;
    this.activeMotion = null;
    this.clipStartedAt = null;
    this.retargeter?.reset();
  }

  _advance(now) {
    this.index += 1;
    if (this.index >= this.queue.length) {
      this.playing = false;
      this.activeMotion = null;
      this.retargeter.reset();
      this.onQueueComplete?.();
      return;
    }

    const entry = this.queue[this.index];
    if (!entry.clip) {
      // Missing motion: skip forward without crashing playback, per the
      // brief's "continue past missing motions" requirement.
      this.onTokenStart?.(entry, 'missing');
      this.onTokenEnd?.(entry, 'missing');
      this._advance(now);
      return;
    }

    this.retargeter.reset();
    this.activeMotion = new BakedAvatarMotion(entry.clip, this.retargeter, {
      easeInSeconds: this.easeInSeconds,
      easeOutSeconds: this.easeOutSeconds,
    });
    this.clipStartedAt = now;
    this.onTokenStart?.(entry, 'ready');
  }

  update(now) {
    if (!this.playing || !this.activeMotion || this.clipStartedAt === null) return;
    const elapsed = now - this.clipStartedAt;
    const sample = this.activeMotion.sample(elapsed);
    if (sample?.ended) {
      const entry = this.queue[this.index];
      this.onTokenEnd?.(entry, 'ready');
      this._advance(now);
    }
  }
}
