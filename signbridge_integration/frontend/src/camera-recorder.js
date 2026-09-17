import { t } from './i18n.js';

/**
 * Record-then-analyze camera workflow (no continuous live recognition).
 * Owns getUserMedia lifecycle and stops every track deterministically.
 *
 * Orientation contract (see docs in frontend for the full audit):
 * - This class never mirrors, flips, or otherwise transforms the actual
 *   camera stream or the MediaRecorder output. The Blob it produces is
 *   exactly what the camera captured, in the same orientation the
 *   recognition backend already expects - callers apply any mirroring as a
 *   CSS `transform` on the <video> preview element only, which is a
 *   display-only change with no effect on the recorded pixel data.
 * - `facingMode` is tracked so a caller can decide whether the preview
 *   should be mirrored (front/"user" cameras conventionally are, for a
 *   natural selfie view; rear/"environment" cameras conventionally are not).
 */
export class CameraRecorder {
  constructor() {
    this.stream = null;
    this.mediaRecorder = null;
    this.chunks = [];
    this.recordedBlob = null;
    this.mimeType = 'video/webm';
    this.facingMode = 'user';
  }

  get isSupported() {
    return Boolean(navigator.mediaDevices?.getUserMedia && window.MediaRecorder);
  }

  /** True when the live preview should be mirrored for a natural selfie view. */
  get shouldMirrorPreview() {
    return this.facingMode !== 'environment';
  }

  async start(videoElement, { facingMode = 'user' } = {}) {
    if (!this.isSupported) {
      throw new Error(t('camera.error.unsupported'));
    }
    this.facingMode = facingMode;
    try {
      // A bare (non-"exact") facingMode is a preference, not a hard
      // requirement - browsers fall back to any available camera if the
      // requested one does not exist, rather than failing outright.
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode },
        audio: false,
      });
    } catch (error) {
      throw this._describeError(error);
    }
    videoElement.srcObject = this.stream;
    await videoElement.play();
  }

  _describeError(error) {
    const messageKeys = {
      NotAllowedError: 'camera.error.permission_denied',
      PermissionDeniedError: 'camera.error.permission_denied',
      NotFoundError: 'camera.error.not_found',
      DevicesNotFoundError: 'camera.error.not_found',
      NotReadableError: 'camera.error.in_use',
      TrackStartError: 'camera.error.in_use',
      OverconstrainedError: 'camera.error.overconstrained',
      SecurityError: 'camera.error.insecure_context',
    };
    const key = messageKeys[error?.name] || 'camera.error.generic';
    const described = new Error(t(key));
    described.cause = error;
    described.cameraErrorName = error?.name || 'UnknownError';
    return described;
  }

  beginRecording() {
    if (!this.stream) throw new Error(t('camera.error.not_started'));
    this.chunks = [];
    const preferredType = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm']
      .find((type) => window.MediaRecorder.isTypeSupported?.(type));
    this.mimeType = preferredType || 'video/webm';

    this.mediaRecorder = new MediaRecorder(this.stream, { mimeType: this.mimeType });
    this.mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) this.chunks.push(event.data);
    };
    this.mediaRecorder.start();
  }

  stopRecording() {
    return new Promise((resolve) => {
      if (!this.mediaRecorder) {
        resolve(null);
        return;
      }
      this.mediaRecorder.onstop = () => {
        this.recordedBlob = new Blob(this.chunks, { type: this.mimeType });
        resolve(this.recordedBlob);
      };
      this.mediaRecorder.stop();
    });
  }

  discardRecording() {
    this.recordedBlob = null;
    this.chunks = [];
  }

  closeCamera() {
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.mediaRecorder = null;
  }
}
