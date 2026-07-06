import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { execFileSync, spawnSync } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { setTimeout as sleep } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';
import { io as createSocket } from 'socket.io-client';

const repoDir = fileURLToPath(new URL('..', import.meta.url));
const corpusFile = `${repoDir}/scripts/fixtures/edge-voice-corpus.json`;
const cli = new Map(process.argv.slice(2).map((value) => {
  const [key, raw = ''] = value.split('=', 2);
  return [key, raw];
}));
const outputDir = cli.get('--output-dir')
  ?? `${repoDir}/plans/260706-0155-audio-playback-tts-fix/reports`;
const summaryJson = cli.get('--summary-json')
  ?? `${outputDir}/phase-05-audio-acceptance.json`;
const evidenceLog = cli.get('--evidence-log')
  ?? `${outputDir}/phase-05-audio-acceptance.log`;
const socketUrl = 'http://127.0.0.1:3030';
const targetEntityId = 'rover-kiwi';
const performanceMetricIntervalMs = 5000;

const corpus = JSON.parse(readFileSync(corpusFile, 'utf8'));
const evidence = [];
const performanceSamples = [];
const edgeVoiceSamples = [];
const trackingSamples = [];
let currentMetricPhase = 'startup';

mkdirSync(outputDir, { recursive: true });

function deepEqual(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function roundConfigFloat(value) {
  return Number(Number(value ?? 0).toFixed(4));
}

function normalizeConfig(config) {
  return {
    language: config.language,
    speaker_id: config.speaker_id,
    speed: roundConfigFloat(config.speed),
    num_steps: config.num_steps,
    volume: roundConfigFloat(config.volume),
  };
}

function sameConfig(a, b) {
  return deepEqual(normalizeConfig(a), normalizeConfig(b));
}

function percentile(values, pct) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil((pct / 100) * sorted.length) - 1));
  return sorted[index];
}

function average(values) {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function summarizePerformanceSamples(samples) {
  const dataflowFpsValues = samples.map((sample) => sample.dataflow_fps).filter((value) => Number.isFinite(value));
  const edgeVoiceCpuValues = samples
    .map((sample) => sample.edge_voice_cpu_percent)
    .filter((value) => Number.isFinite(value));
  const edgeVoiceRssValues = samples
    .map((sample) => sample.edge_voice_memory_mb)
    .filter((value) => Number.isFinite(value));
  return {
    samples: samples.length,
    avg_dataflow_fps: Number(average(dataflowFpsValues).toFixed(2)),
    p95_dataflow_fps: Number(percentile(dataflowFpsValues, 95).toFixed(2)),
    avg_edge_voice_cpu_percent: Number(average(edgeVoiceCpuValues).toFixed(2)),
    peak_edge_voice_cpu_percent: Number(percentile(edgeVoiceCpuValues, 100).toFixed(1)),
    peak_edge_voice_rss_mb: Number(percentile(edgeVoiceRssValues, 100).toFixed(1)),
  };
}

function metricsForPhase(phase) {
  return performanceSamples.filter((sample) => sample.phase === phase);
}

function setMetricPhase(phase) {
  currentMetricPhase = phase;
  evidence.push(`metric_phase=${phase}`);
}

function estimateAudioMs(text, language, config) {
  const normalized = text.replace(/\s+/g, ' ').trim();
  const chars = normalized.length;
  const charsPerSecond = language === 'vi' ? 14 : 16;
  const stepFactor = 1 + Math.max(0, config.num_steps - 8) * 0.035;
  const speedFactor = config.speed > 0 ? 1 / config.speed : 1;
  return Math.max(400, Math.round((chars / charsPerSecond) * 1000 * stepFactor * speedFactor));
}

function runCommand(command, args, timeoutMs = 15000) {
  try {
    return execFileSync(command, args, {
      encoding: 'utf8',
      maxBuffer: 10 * 1024 * 1024,
      timeout: timeoutMs,
    });
  } catch (error) {
    if (typeof error.stdout === 'string' && error.stdout.length > 0) return error.stdout;
    if (Buffer.isBuffer(error.stdout) && error.stdout.length > 0) return error.stdout.toString('utf8');
    throw error;
  }
}

function runLogCommand(command, args, timeoutMs = 15000) {
  const result = spawnSync(command, args, {
    encoding: 'utf8',
    maxBuffer: 10 * 1024 * 1024,
    timeout: timeoutMs,
  });
  const output = `${result.stdout ?? ''}${result.stderr ?? ''}`;
  if (result.error && !output) {
    throw result.error;
  }
  if (result.status !== 0 && !output) {
    throw new Error(`${command} ${args.join(' ')} failed with status ${result.status}`);
  }
  return output;
}

function parseLatestInteger(output, pattern) {
  const matches = [...output.matchAll(pattern)];
  if (!matches.length) return null;
  return Number.parseInt(matches[matches.length - 1][1], 10);
}

function buildConfig(baseConfig, override) {
  return {
    language: override.language ?? baseConfig.language,
    speaker_id: override.speaker_id ?? baseConfig.speaker_id,
    speed: override.speed ?? baseConfig.speed,
    num_steps: override.num_steps ?? baseConfig.num_steps,
    volume: override.volume ?? baseConfig.volume,
  };
}

function makeWalkieFrame(streamId, frameIndex, frameSize, amplitude, sampleRate = 48_000) {
  const samples = new Float32Array(frameSize);
  for (let i = 0; i < frameSize; i += 1) {
    const phase = ((frameIndex * frameSize) + i) / sampleRate;
    samples[i] = amplitude * Math.sin(2 * Math.PI * 220 * phase);
  }
  return {
    metadata: {
      protocol_version: 1,
      stream_id: streamId,
      frame_id: frameIndex,
      capture_timestamp_ms: Date.now(),
      sample_rate: sampleRate,
      channels: 1,
      sample_count: samples.length,
      format: 'f32le',
    },
    samples,
  };
}

class SocketSession {
  constructor(url, auth) {
    this.url = url;
    this.auth = auth;
    this.socket = null;
    this.connected = false;
    this.events = [];
    this.waiters = [];
    this.latestPerformanceMetrics = null;
    this.latestTrackingTelemetry = null;
  }

  async connect(timeoutMs = 15000) {
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error(`socket connect timeout after ${timeoutMs}ms`)), timeoutMs);
      this.socket = createSocket(this.url, {
        path: '/socket.io/',
        transports: ['websocket'],
        reconnection: false,
        forceNew: true,
        autoConnect: false,
        auth: this.auth,
      });
      this.socket.onAny((event, payload) => {
        this._dispatch(event, payload);
      });
      this.socket.on('connect', () => {
        this.connected = true;
        clearTimeout(timeout);
        resolve();
      });
      this.socket.on('connect_error', (error) => {
        clearTimeout(timeout);
        reject(new Error(`socket connect error: ${error.message}`));
      });
      this.socket.connect();
    });
  }

  close() {
    if (this.socket) {
      this.socket.removeAllListeners();
      this.socket.close();
      this.socket = null;
    }
  }

  emit(event, ...args) {
    if (!this.connected || !this.socket) throw new Error(`cannot emit ${event} before socket connection`);
    if (event === 'audio_stream' && args.length !== 2) {
      throw new Error(`audio_stream requires metadata plus one binary attachment, got ${args.length} args`);
    }
    this.socket.emit(event, ...args);
  }

  async waitFor(event, predicate = () => true, timeoutMs = 15000) {
    const existingIndex = this.events.findIndex((entry) => entry.event === event && predicate(entry.payload));
    if (existingIndex !== -1) return this.events.splice(existingIndex, 1)[0];

    return await new Promise((resolve, reject) => {
      const waiter = {
        event,
        predicate,
        resolve,
        reject,
        timer: setTimeout(() => {
          this.waiters = this.waiters.filter((item) => item !== waiter);
          reject(new Error(`timed out waiting for ${event} after ${timeoutMs}ms`));
        }, timeoutMs),
      };
      this.waiters.push(waiter);
    });
  }

  _dispatch(event, payload) {
    if (event === 'audio_frame' || event === 'video_frame') {
      return;
    }
    const item = { event, payload, receivedAt: Date.now() };
    if (event === 'performance_metrics') {
      this.latestPerformanceMetrics = payload;
      const edgeVoice = payload?.node_metrics?.['edge-voice'];
      performanceSamples.push({
        at: item.receivedAt,
        phase: currentMetricPhase,
        dataflow_fps: payload.dataflow_fps,
        total_cpu_percent: payload.total_cpu_percent,
        edge_voice_found: Boolean(edgeVoice),
        edge_voice_cpu_percent: edgeVoice?.cpu_usage_percent ?? null,
        edge_voice_memory_mb: edgeVoice?.memory_usage_mb ?? null,
        edge_voice_fps: edgeVoice?.fps ?? null,
      });
      if (edgeVoice) {
        edgeVoiceSamples.push(edgeVoice);
      }
    }

    if (event === 'tracking_telemetry') {
      this.latestTrackingTelemetry = payload;
      trackingSamples.push({
        at: item.receivedAt,
        phase: currentMetricPhase,
        state: payload?.state ?? 'unknown',
      });
    }

    const waiterIndex = this.waiters.findIndex((waiter) => waiter.event === event && waiter.predicate(payload));
    if (waiterIndex !== -1) {
      const waiter = this.waiters.splice(waiterIndex, 1)[0];
      clearTimeout(waiter.timer);
      waiter.resolve(item);
      return;
    }

    this.events.push(item);
  }
}

async function sendTrackingCommand(client, commandType, expectedStates, timeoutMs = 30000) {
  client.emit('tracking_command', { command_type: commandType });
  const telemetry = await client.waitFor(
    'tracking_telemetry',
    (value) => expectedStates.includes(value.state),
    timeoutMs,
  );
  evidence.push(`tracking_command=${commandType} state=${telemetry.payload.state}`);
  return telemetry.payload;
}

async function ensureVisionActive(client) {
  const detection = await sendTrackingCommand(
    client,
    'enable_detection',
    ['DetectionOnly', 'Enabled', 'Tracking', 'TargetLost'],
  );
  if (detection.state === 'DetectionOnly') {
    return await sendTrackingCommand(client, 'enable', ['Enabled', 'Tracking', 'TargetLost']);
  }
  return detection;
}

async function disableVision(client) {
  return await sendTrackingCommand(client, 'disable_detection', ['Disabled']);
}

async function samplePerformanceWindow(phase, durationMs, minSamples) {
  const startIndex = performanceSamples.length;
  setMetricPhase(phase);
  const deadline = Date.now() + Math.max(durationMs, (minSamples * performanceMetricIntervalMs) + 2000);
  let samples = [];
  do {
    await sleep(500);
    samples = performanceSamples.slice(startIndex).filter((sample) => sample.phase === phase);
  } while (samples.length < minSamples && Date.now() < deadline);
  if (samples.length < minSamples) {
    throw new Error(`expected at least ${minSamples} performance samples for ${phase}, got ${samples.length}`);
  }
  const summary = summarizePerformanceSamples(samples);
  evidence.push(
    `performance_window=${phase} samples=${summary.samples} avg_dataflow_fps=${summary.avg_dataflow_fps} peak_edge_voice_rss_mb=${summary.peak_edge_voice_rss_mb}`,
  );
  return summary;
}

async function collectDefaultState(client, defaultConfig) {
  let ttsConfig;
  try {
    ttsConfig = await client.waitFor('tts_config_state', () => true, 2000);
  } catch {
    client.emit('tts_config_update', { base_revision: 9_999, config: defaultConfig });
    ttsConfig = await client.waitFor('tts_config_state', () => true, 15000);
  }

  let currentRevision = ttsConfig.payload.desired_revision;
  if (!sameConfig(ttsConfig.payload.desired_config, defaultConfig)) {
    client.emit('tts_config_update', { base_revision: currentRevision, config: defaultConfig });
    ttsConfig = await client.waitFor(
      'tts_config_state',
      (value) => value.desired_revision === currentRevision + 1 && sameConfig(value.desired_config, defaultConfig),
      15000,
    );
    currentRevision = ttsConfig.payload.desired_revision;
  }

  const ready = await client.waitFor(
    'voice_status',
    (status) => status.state === 'ready' && status.applied_revision === currentRevision,
    180000,
  );

  evidence.push(`default_revision=${currentRevision} desired_config=${JSON.stringify(ttsConfig.payload.desired_config)}`);
  evidence.push(`voice_ready_revision=${ready.payload.applied_revision}`);
  return { revision: currentRevision, config: ttsConfig.payload.desired_config };
}

async function updateConfig(client, currentRevision, config) {
  const nextRevision = currentRevision + 1;
  client.emit('tts_config_update', { base_revision: currentRevision, config });

  const state = await client.waitFor(
    'tts_config_state',
    (value) => value.desired_revision === nextRevision && sameConfig(value.desired_config, config),
    15000,
  );

  const voice = await client.waitFor(
    'voice_status',
    (value) => value.applied_revision === nextRevision && sameConfig(value.applied_config, config) && value.state === 'ready',
    180000,
  );

  evidence.push(`config_revision=${nextRevision} speaker=${config.speaker_id} language=${config.language} ready`);
  return { revision: nextRevision, state: state.payload, voice: voice.payload };
}

async function runTtsCase(client, currentRevision, testCase, label) {
  const config = buildConfig(corpus.default_config, testCase);
  const updated = await updateConfig(client, currentRevision, config);
  const commandStart = Date.now();

  client.emit('tts_command', { text: testCase.text });
  const ack = await client.waitFor(
    'tts_command_ack',
    (value) => value.state === 'accepted' && value.target_entity_id === targetEntityId,
    10000,
  );
  const speaking = await client.waitFor(
    'voice_status',
    (value) => value.active_command_id === ack.payload.command_id && value.state === 'speaking',
    30000,
  );
  const result = await client.waitFor(
    'tts_command_result',
    (value) => value.command_id === ack.payload.command_id,
    180000,
  );

  if (result.payload.state !== 'completed') {
    throw new Error(`${label} expected completed result, got ${result.payload.state}`);
  }

  const ttfaMs = speaking.receivedAt - commandStart;
  const wallMs = result.receivedAt - commandStart;
  const estimatedAudioMs = estimateAudioMs(testCase.text, config.language, config);
  const estimatedRtf = wallMs / estimatedAudioMs;

  evidence.push(
    `${label} cmd=${ack.payload.command_id} ttfa_ms=${ttfaMs} wall_ms=${wallMs} estimated_rtf=${estimatedRtf.toFixed(3)} edge_voice_rss_mb=${(
      client.latestPerformanceMetrics?.node_metrics?.['edge-voice']?.memory_usage_mb ?? 0
    ).toFixed(1)}`,
  );

  return {
    revision: updated.revision,
    label,
    command_id: ack.payload.command_id,
    language: config.language,
    speaker_id: config.speaker_id,
    state: result.payload.state,
    ttfa_ms: ttfaMs,
    wall_ms: wallMs,
    estimated_audio_ms: estimatedAudioMs,
    estimated_rtf: Number(estimatedRtf.toFixed(3)),
    edge_voice_cpu_percent: client.latestPerformanceMetrics?.node_metrics?.['edge-voice']?.cpu_usage_percent ?? 0,
    edge_voice_memory_mb: client.latestPerformanceMetrics?.node_metrics?.['edge-voice']?.memory_usage_mb ?? 0,
  };
}

async function runWalkiePreemption(client, currentRevision) {
  const walkieCase = corpus.walkie;
  const config = buildConfig(corpus.default_config, walkieCase);
  const updated = await updateConfig(client, currentRevision, config);
  const commandStart = Date.now();
  const streamId = randomUUID();

  client.emit('tts_command', { text: walkieCase.text });
  const ack = await client.waitFor(
    'tts_command_ack',
    (value) => value.state === 'accepted' && value.target_entity_id === targetEntityId,
    10000,
  );

  await client.waitFor(
    'voice_status',
    (value) => value.active_command_id === ack.payload.command_id && value.state === 'speaking',
    30000,
  );

  await sleep(walkieCase.preemption_delay_ms ?? 150);
  for (let index = 0; index < (walkieCase.frame_count ?? 8); index += 1) {
    const frame = makeWalkieFrame(
      streamId,
      index,
      walkieCase.frame_size ?? 512,
      walkieCase.amplitude ?? 0.2,
      walkieCase.sample_rate ?? 48_000,
    );
    client.emit('audio_stream', frame.metadata, frame.samples);
    await sleep(walkieCase.frame_spacing_ms ?? 20);
  }

  const result = await client.waitFor(
    'tts_command_result',
    (value) => value.command_id === ack.payload.command_id,
    180000,
  );
  if (result.payload.state !== 'interrupted') {
    throw new Error(`walkie preemption expected interrupted result, got ${result.payload.state}`);
  }

  const wallMs = result.receivedAt - commandStart;
  evidence.push(`walkie cmd=${ack.payload.command_id} result=${result.payload.state} wall_ms=${wallMs}`);

  return {
    revision: updated.revision,
    wall_ms: wallMs,
    command_id: ack.payload.command_id,
    result_state: result.payload.state,
    reason_code: result.payload.reason_code ?? null,
  };
}

async function runSoak(client, currentRevision) {
  const soak = corpus.soak;
  const config = buildConfig(corpus.default_config, soak);
  const updated = await updateConfig(client, currentRevision, config);
  const count = soak.count ?? 100;
  const ttfaValues = [];
  const rtfValues = [];

  for (let index = 0; index < count; index += 1) {
    const text = `${soak.prefix ?? soak.text} ${String(index + 1).padStart(3, '0')}`;
    const start = Date.now();
    client.emit('tts_command', { text });
    const ack = await client.waitFor(
      'tts_command_ack',
      (value) => value.state === 'accepted' && value.target_entity_id === targetEntityId,
      10000,
    );
    const speaking = await client.waitFor(
      'voice_status',
      (value) => value.active_command_id === ack.payload.command_id && value.state === 'speaking',
      30000,
    );
    const result = await client.waitFor(
      'tts_command_result',
      (value) => value.command_id === ack.payload.command_id,
      180000,
    );
    if (result.payload.state !== 'completed') {
      throw new Error(`soak command ${index + 1} expected completed result, got ${result.payload.state}`);
    }

    const ttfaMs = speaking.receivedAt - start;
    const wallMs = result.receivedAt - start;
    const estimatedAudioMs = estimateAudioMs(text, config.language, config);
    const estimatedRtf = wallMs / estimatedAudioMs;
    ttfaValues.push(ttfaMs);
    rtfValues.push(estimatedRtf);

    if ((index + 1) % 10 === 0) {
      evidence.push(`soak_progress=${index + 1}/${count} p95_ttfa_ms=${percentile(ttfaValues, 95).toFixed(1)} p95_estimated_rtf=${percentile(rtfValues, 95).toFixed(3)}`);
    }
  }

  const summary = {
    revision: updated.revision,
    count,
    ttfa_values_ms: ttfaValues,
    estimated_rtf_values: rtfValues,
    p95_ttfa_ms: percentile(ttfaValues, 95),
    p95_estimated_rtf: percentile(rtfValues, 95),
  };

  evidence.push(`soak_done count=${count} p95_ttfa_ms=${summary.p95_ttfa_ms.toFixed(1)} p95_estimated_rtf=${summary.p95_estimated_rtf.toFixed(3)}`);
  return summary;
}

async function collectCaptureSuppression(startedAt) {
  let roverContainer = null;
  try {
    const runningContainers = runCommand('docker', ['ps', '--format', '{{.Names}}'], 10000)
      .split('\n')
      .map((value) => value.trim())
      .filter(Boolean);
    roverContainer = runningContainers.find((value) => /rover-kiwi/i.test(value)) ?? null;
  } catch (error) {
    evidence.push(`capture_log_source_probe_failed=${error.message}`);
  }
  const output = roverContainer
    ? runLogCommand('docker', ['logs', '--since', startedAt, roverContainer], 20000)
    : runLogCommand('timeout', ['15', 'dora', 'logs', 'rover-kiwi', 'audio-capture'], 20000);
  const samplesRejected = parseLatestInteger(output, /samples_rejected=(\d+)/g);
  const captureDrops = parseLatestInteger(output, /drops=(\d+)/g);
  const effectiveSamplesRejected = samplesRejected ?? captureDrops;
  evidence.push(`capture_log_source=${roverContainer ? `docker:${roverContainer}` : 'dora'}`);
  evidence.push(`capture_samples_rejected=${effectiveSamplesRejected ?? 'unavailable'} capture_drops=${captureDrops ?? 'unavailable'}`);
  return { samplesRejected: effectiveSamplesRejected, captureDrops, raw: output };
}

async function main() {
  runCommand('curl', ['-fsS', 'http://127.0.0.1:3030/health'], 5000);

  const client = new SocketSession(socketUrl, {
    username: 'admin',
    password: 'password',
    token: null,
  });

  const startedAt = new Date().toISOString();
  await client.connect(15000);
  evidence.push(`connected socket_url=${socketUrl}`);

  setMetricPhase('initial_state');
  const defaults = await collectDefaultState(client, corpus.default_config);
  const baselineMetrics = await client.waitFor('performance_metrics', () => true, 15000);
  evidence.push(`baseline_dataflow_fps=${baselineMetrics.payload.dataflow_fps} baseline_edge_voice_rss_mb=${baselineMetrics.payload.node_metrics?.['edge-voice']?.memory_usage_mb ?? 0}`);

  setMetricPhase('vision_activation');
  const visionState = await ensureVisionActive(client);
  evidence.push(`vision_state=${visionState.state}`);

  const visionOnly = await samplePerformanceWindow('vision_only', 5000, 2);

  const balancedResults = [];
  let revision = defaults.revision;
  setMetricPhase('vision_tts');
  for (const [index, testCase] of corpus.balanced_cases.entries()) {
    const result = await runTtsCase(client, revision, testCase, testCase.id ?? `balanced-${index + 1}`);
    revision += 1;
    balancedResults.push(result);
  }

  const walkie = await runWalkiePreemption(client, revision);
  revision = walkie.revision;

  const restoredDefaults = await updateConfig(client, revision, corpus.default_config);
  revision = restoredDefaults.revision;

  const soak = await runSoak(client, revision);

  await sleep(500);
  const capture = await collectCaptureSuppression(startedAt);
  setMetricPhase('vision_shutdown');
  const disabledVision = await disableVision(client);
  evidence.push(`vision_state=${disabledVision.state}`);
  const latestMetrics = client.latestPerformanceMetrics;
  const visionTtsSamples = metricsForPhase('vision_tts');
  if (!visionTtsSamples.length) {
    throw new Error('no performance samples recorded while vision and TTS were active');
  }
  if (!edgeVoiceSamples.length) {
    throw new Error('no edge-voice performance samples were observed');
  }
  const visionTts = summarizePerformanceSamples(visionTtsSamples);
  const peakEdgeVoiceCpu = percentile(edgeVoiceSamples.map((sample) => sample.cpu_usage_percent), 100);
  const peakEdgeVoiceRss = percentile(edgeVoiceSamples.map((sample) => sample.memory_usage_mb), 100);
  const peakDataflowFps = percentile(performanceSamples.map((sample) => sample.dataflow_fps), 100);
  const visionFpsRegressionPercent = visionOnly.avg_dataflow_fps > 0
    ? Math.max(0, ((visionOnly.avg_dataflow_fps - visionTts.avg_dataflow_fps) / visionOnly.avg_dataflow_fps) * 100)
    : 0;
  const summary = {
    started_at_utc: startedAt,
    socket_url: socketUrl,
    target_entity_id: targetEntityId,
    corpus_file: corpusFile,
    default_revision: defaults.revision,
    balanced_cases: balancedResults.length,
    soak_count: soak.count,
    results: {
      balanced: balancedResults,
      walkie,
      soak,
    },
    metrics: {
      samples: performanceSamples.length,
      peak_edge_voice_cpu_percent: Number(peakEdgeVoiceCpu.toFixed(1)),
      peak_edge_voice_rss_mb: Number(peakEdgeVoiceRss.toFixed(1)),
      peak_dataflow_fps: Number(peakDataflowFps.toFixed(1)),
      phases: {
        vision_only: visionOnly,
        vision_tts: visionTts,
      },
      latest_total_cpu_percent: latestMetrics?.total_cpu_percent ?? 0,
      latest_total_memory_mb: latestMetrics?.total_memory_mb ?? 0,
      latest_edge_voice_cpu_percent: latestMetrics?.node_metrics?.['edge-voice']?.cpu_usage_percent ?? 0,
      latest_edge_voice_memory_mb: latestMetrics?.node_metrics?.['edge-voice']?.memory_usage_mb ?? 0,
    },
    capture: {
      samples_rejected: capture.samplesRejected,
      drops: capture.captureDrops,
    },
    thresholds: {
      ttfa_p95_ms: soak.p95_ttfa_ms,
      estimated_rtf_p95: soak.p95_estimated_rtf,
      peak_edge_voice_rss_mb: peakEdgeVoiceRss,
      vision_fps_regression_percent: Number(visionFpsRegressionPercent.toFixed(2)),
    },
  };

  const failures = [];
  if (visionOnly.samples < 2) failures.push(`expected at least 2 vision-only metric samples, got ${visionOnly.samples}`);
  if (visionTts.samples < 1) failures.push(`expected at least 1 concurrent vision+tts metric sample, got ${visionTts.samples}`);
  if (soak.p95_ttfa_ms >= 1000) failures.push(`p95 TTFA ${soak.p95_ttfa_ms.toFixed(1)}ms >= 1000ms`);
  if (soak.p95_estimated_rtf >= 1.0) failures.push(`p95 estimated RTF ${soak.p95_estimated_rtf.toFixed(3)} >= 1.0`);
  if (peakEdgeVoiceRss >= 2048) failures.push(`peak edge_voice RSS ${peakEdgeVoiceRss.toFixed(1)}MB >= 2048MB`);
  if (visionFpsRegressionPercent > 10) failures.push(`vision FPS regression ${visionFpsRegressionPercent.toFixed(2)}% > 10%`);
  if ((capture.samplesRejected ?? 0) <= 0) failures.push(`expected audio capture suppression counter to be > 0, got ${capture.samplesRejected ?? 'unavailable'}`);
  if ((capture.captureDrops ?? 0) <= 0) failures.push(`expected audio capture drops to be > 0, got ${capture.captureDrops ?? 'unavailable'}`);

  writeFileSync(summaryJson, `${JSON.stringify(summary, null, 2)}\n`);
  writeFileSync(evidenceLog, `${evidence.join('\n')}\n`);

  console.log(`summary_json=${summaryJson}`);
  console.log(`evidence_log=${evidenceLog}`);
  console.log(`cases=${balancedResults.length + 1 + soak.count}`);
  console.log(`p95_ttfa_ms=${soak.p95_ttfa_ms.toFixed(1)}`);
  console.log(`p95_estimated_rtf=${soak.p95_estimated_rtf.toFixed(3)}`);
  console.log(`peak_edge_voice_rss_mb=${peakEdgeVoiceRss.toFixed(1)}`);
  console.log(`vision_fps_regression_percent=${visionFpsRegressionPercent.toFixed(2)}`);
  console.log(`capture_samples_rejected=${capture.samplesRejected ?? 'unavailable'}`);

  if (failures.length > 0) {
    throw new Error(failures.join('; '));
  }
}

await main();
