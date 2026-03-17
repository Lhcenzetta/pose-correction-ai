'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { useParams, useRouter } from 'next/navigation';

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface SessionConfig {
  session_id: number;
  exercise_id: number;
  exercise_name: string;
  duration_seconds: number;
}

interface PoseResult {
  confidence: number;
  is_correct: boolean;
  tip: string;
  left_abduction: number;
  right_abduction: number;
  accuracy_score: number;
}

export default function SessionPage() {
  const router = useRouter();
  const params = useParams();
  const sessionId = params?.id ? Number(params.id) : null;

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const poseRef = useRef<any>(null);
  const cameraRef = useRef<any>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const isSendingRef = useRef(false);
  const sessionRef = useRef<SessionConfig | null>(null);
  const scoresRef = useRef<number[]>([]);

  const [config, setConfig] = useState<SessionConfig | null>(null);
  const [timeLeft, setTimeLeft] = useState(0);
  const [result, setResult] = useState<PoseResult | null>(null);
  const [pageStatus, setPageStatus] = useState<'loading' | 'ready' | 'running'>('loading');
  const [error, setError] = useState('');

  // ── Load external scripts ──────────────────────────────────────
  function loadScript(src: string): Promise<void> {
    return new Promise((resolve, reject) => {
      if (document.querySelector(`script[src="${src}"]`)) { resolve(); return; }
      const s = document.createElement('script');
      s.src = src;
      s.crossOrigin = 'anonymous';
      s.onload = () => resolve();
      s.onerror = () => reject(new Error(`Failed to load ${src}`));
      document.head.appendChild(s);
    });
  }

  // ── Finalize → redirect to results ────────────────────────────
  const finalizeSession = useCallback(async (sid: number) => {
    const token = localStorage.getItem('access_token');
    try {
      await fetch(`${API}/session/${sid}/finalize`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      });
    } catch { /* ignore network errors — session already saved */ }

    if (timerRef.current) clearInterval(timerRef.current);
    if (cameraRef.current) cameraRef.current.stop();

    router.push('/results');
  }, [router]);

  // ── Send pose frame to API ─────────────────────────────────────
  const sendPose = useCallback(async (features: number[]) => {
    if (isSendingRef.current || !sessionRef.current) return;
    isSendingRef.current = true;

    const token = localStorage.getItem('access_token');
    try {
      const res = await fetch(`${API}/process-pose`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          session_id: Number(sessionRef.current.session_id),
          features,
        }),
      });

      if (res.ok) {
        const data: PoseResult = await res.json();
        setResult(data);
        scoresRef.current.push(data.accuracy_score);
      } else {
        const err = await res.json();
        console.error('process-pose error:', err);
      }
    } catch (e) {
      console.error('sendPose failed:', e);
    } finally {
      isSendingRef.current = false;
    }
  }, []);

  // ── Feature extraction — matches backend exactly ───────────────
  function calcAngle(p1: any, p2: any, p3: any): number {
    const v1 = { x: p1.x - p2.x, y: p1.y - p2.y };
    const v2 = { x: p3.x - p2.x, y: p3.y - p2.y };
    const dot = v1.x * v2.x + v1.y * v2.y;
    const mag1 = Math.sqrt(v1.x ** 2 + v1.y ** 2);
    const mag2 = Math.sqrt(v2.x ** 2 + v2.y ** 2);
    return (Math.acos(Math.max(-1, Math.min(1, dot / (mag1 * mag2 + 1e-6)))) * 180) / Math.PI;
  }

  function extractFeatures(landmarks: any[]): number[] {
    const upperBodyIds = [11, 12, 13, 14, 15, 16, 23, 24];
    const features: number[] = [];

    // 24 coordinate features
    upperBodyIds.forEach(id => {
      const lm = landmarks[id];
      features.push(lm.x, lm.y, lm.z);
    });

    // 4 angle features
    features.push(
      calcAngle(landmarks[11], landmarks[13], landmarks[15]), // left elbow
      calcAngle(landmarks[12], landmarks[14], landmarks[16]), // right elbow
      calcAngle(landmarks[23], landmarks[11], landmarks[13]), // left abduction
      calcAngle(landmarks[24], landmarks[12], landmarks[14]), // right abduction
    );

    return features; // exactly 28
  }

  // ── MediaPipe callback ─────────────────────────────────────────
  const onResults = useCallback((results: any) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;

    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

    if (results.poseLandmarks) {
      const w = window as any;
      if (w.drawConnectors && w.POSE_CONNECTIONS) {
        w.drawConnectors(ctx, results.poseLandmarks, w.POSE_CONNECTIONS, {
          color: 'rgba(255,255,255,0.35)',
          lineWidth: 2,
        });
      }
      if (w.drawLandmarks) {
        w.drawLandmarks(ctx, results.poseLandmarks, {
          color: '#4fffb0',
          lineWidth: 1,
          radius: 4,
        });
      }

      // Only send to API when session is running
      if (pageStatus === 'running') {
        const features = extractFeatures(results.poseLandmarks);
        sendPose(features);
      }
    }
    ctx.restore();
  }, [pageStatus, sendPose]);

  // ── Init: load config + MediaPipe ─────────────────────────────
  useEffect(() => {
    async function init() {
      const token = localStorage.getItem('access_token');
      if (!token) { router.push('/login'); return; }

      let active: SessionConfig | null = null;
      const raw = localStorage.getItem('active_session');
      
      if (raw) {
        const parsed: SessionConfig = JSON.parse(raw);
        if (sessionId && parsed.session_id === sessionId) {
          active = parsed;
        }
      }

      // Fallback: Fetch from API if sessionId mismatch or direct link
      if (!active && sessionId) {
        try {
          // get user first to list sessions
          const meRes = await fetch(`${API}/me`, { headers: { Authorization: `Bearer ${token}` } });
          if (meRes.ok) {
            const me = await meRes.json();
            const sessRes = await fetch(`${API}/sessions/user/${me.id}`, { headers: { Authorization: `Bearer ${token}` } });
            if (sessRes.ok) {
              const sessions: any[] = await sessRes.json();
              const found = sessions.find(s => s.id === sessionId);
              if (found) {
                active = {
                  session_id: found.id,
                  exercise_id: found.exercise_id,
                  exercise_name: found.exercise_name,
                  duration_seconds: found.duration_seconds
                };
              }
            }
          }
        } catch (err) {
          console.error('Failed to fetch session fallback:', err);
        }
      }

      if (!active) {
        router.push('/select-exercise');
        return;
      }

      setConfig(active);
      sessionRef.current = active;
      setTimeLeft(active.duration_seconds);

      try {
        await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/pose/pose.js');
        await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js');
        await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js');

        const w = window as any;

        const pose = new w.Pose({
          locateFile: (file: string) =>
            `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
        });

        pose.setOptions({
          modelComplexity: 1,
          smoothLandmarks: true,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        pose.onResults(onResults);
        poseRef.current = pose;

        const camera = new w.Camera(videoRef.current, {
          onFrame: async () => {
            await pose.send({ image: videoRef.current });
          },
          width: 640,
          height: 480,
        });

        cameraRef.current = camera;
        await camera.start();
        setPageStatus('ready');
      } catch (e) {
        setError('Could not load camera or MediaPipe. Check your internet connection.');
        console.error(e);
      }
    }

    init();

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (cameraRef.current) cameraRef.current.stop();
    };
  }, [sessionId, onResults, router]);

  // ── Keep onResults fresh when pageStatus changes ───────────────
  useEffect(() => {
    if (poseRef.current) poseRef.current.onResults(onResults);
  }, [onResults]);

  // ── Start countdown ────────────────────────────────────────────
  function startSession() {
    setPageStatus('running');

    timerRef.current = setInterval(() => {
      setTimeLeft(prev => {
        if (prev <= 1) {
          clearInterval(timerRef.current!);
          finalizeSession(sessionRef.current!.session_id);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  }

  // ── Format mm:ss ───────────────────────────────────────────────
  function fmtTime(s: number) {
    return `${Math.floor(s / 60).toString().padStart(2, '0')}:${(s % 60).toString().padStart(2, '0')}`;
  }

  const totalSeconds = config?.duration_seconds || 1;
  const progress     = ((totalSeconds - timeLeft) / totalSeconds) * 100;

  // ── Styles ─────────────────────────────────────────────────────
  const S: Record<string, React.CSSProperties> = {
    page: {
      fontFamily: "'DM Sans', sans-serif",
      background: '#f7f9f7',
      minHeight: '100vh',
    },
    nav: {
      background: '#fff',
      borderBottom: '1px solid #e3ede5',
      padding: '1rem 2rem',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      position: 'sticky',
      top: 0,
      zIndex: 10,
    },
    main: {
      maxWidth: 1020,
      margin: '0 auto',
      padding: '2rem',
      display: 'grid',
      gridTemplateColumns: '1fr 310px',
      gap: '1.5rem',
      alignItems: 'start',
    },
    videoWrap: {
      background: '#0f1f13',
      borderRadius: 20,
      overflow: 'hidden',
      position: 'relative',
      aspectRatio: '4/3',
    },
    sidebar: {
      display: 'flex',
      flexDirection: 'column',
      gap: '1rem',
    },
    card: {
      background: '#fff',
      border: '1px solid #e3ede5',
      borderRadius: 16,
      padding: '1.25rem 1.5rem',
    },
    sectionLabel: {
      fontSize: 11,
      textTransform: 'uppercase' as const,
      letterSpacing: '0.08em',
      color: '#8aaa90',
      marginBottom: 8,
    },
  };

  // ── Render ─────────────────────────────────────────────────────
  return (
    <div style={S.page}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,300;0,400;0,600;1,300&family=DM+Sans:wght@300;400;500&display=swap');
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.4; } }
        @media (max-width: 700px) {
          .sess-main { grid-template-columns: 1fr !important; }
        }
      `}</style>

      {/* Navbar */}
      <nav style={S.nav}>
        <div style={{ fontFamily: "'Fraunces', serif", fontWeight: 600, fontSize: 18, color: '#1a6640' }}>
          Pose<span style={{ color: '#b5cfb9' }}>Correct</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <span style={{ fontSize: 12, color: '#8aaa90' }}>{config?.exercise_name}</span>
          {pageStatus === 'running' && (
            <span style={{ fontSize: 12, background: '#e8f4ec', color: '#1a6640', padding: '4px 12px', borderRadius: 100, fontWeight: 500, animation: 'pulse 2s infinite' }}>
              ● Live
            </span>
          )}
        </div>
      </nav>

      {/* Main grid */}
      <main className="sess-main" style={S.main}>

        {/* ── Webcam panel ── */}
        <div style={S.videoWrap}>
          <video ref={videoRef} style={{ display: 'none' }} />
          <canvas
            ref={canvasRef}
            width={640}
            height={480}
            style={{ width: '100%', height: '100%', display: 'block' }}
          />

          {/* Loading / ready overlay */}
          {pageStatus !== 'running' && (
            <div style={{ position: 'absolute', inset: 0, background: 'rgba(15,31,19,0.75)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: 14 }}>
              {pageStatus === 'loading' && (
                <>
                  <div style={{ width: 34, height: 34, border: '2px solid rgba(79,255,176,0.25)', borderTopColor: '#4fffb0', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
                  <span style={{ color: '#b5cfb9', fontSize: 13, fontWeight: 300 }}>Loading camera...</span>
                </>
              )}
              {pageStatus === 'ready' && (
                <>
                  <div style={{ fontSize: '2.5rem' }}>📷</div>
                  <span style={{ color: '#e8f2ec', fontSize: 15, fontWeight: 500 }}>Camera ready</span>
                  <span style={{ color: '#8aaa90', fontSize: 12, fontWeight: 300 }}>Stand in frame, then press Start</span>
                </>
              )}
            </div>
          )}

          {/* Timer badge on canvas */}
          {pageStatus === 'running' && (
            <div style={{ position: 'absolute', top: 14, left: 14, background: 'rgba(15,31,19,0.8)', borderRadius: 12, padding: '6px 16px', backdropFilter: 'blur(4px)' }}>
              <span style={{ fontFamily: "'Fraunces', serif", fontSize: '1.6rem', fontWeight: 600, color: timeLeft <= 10 ? '#ffaa00' : '#4fffb0', letterSpacing: -1 }}>
                {fmtTime(timeLeft)}
              </span>
            </div>
          )}

          {/* Live feedback bar on canvas */}
          {pageStatus === 'running' && result && (
            <div style={{ position: 'absolute', bottom: 14, left: 14, right: 14, background: result.is_correct ? 'rgba(26,102,64,0.88)' : 'rgba(160,40,20,0.88)', borderRadius: 12, padding: '10px 16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', backdropFilter: 'blur(4px)' }}>
              <span style={{ fontSize: 13, color: '#fff', fontWeight: 500 }}>{result.tip}</span>
              <span style={{ fontSize: 13, color: result.is_correct ? '#9de8c0' : '#ffaa88', fontWeight: 600, fontFamily: "'Fraunces', serif" }}>
                {Math.round(result.accuracy_score)}%
              </span>
            </div>
          )}
        </div>

        {/* ── Sidebar ── */}
        <div style={S.sidebar}>

          {/* Timer card */}
          <div style={S.card}>
            <div style={S.sectionLabel}>Time remaining</div>
            <div style={{ fontFamily: "'Fraunces', serif", fontSize: '3rem', fontWeight: 600, color: timeLeft <= 10 && pageStatus === 'running' ? '#b06a00' : '#0f1f13', letterSpacing: -2, lineHeight: 1, textAlign: 'center' }}>
              {fmtTime(timeLeft)}
            </div>
            <div style={{ marginTop: '1rem', height: 5, background: '#e8f4ec', borderRadius: 100, overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${progress}%`, background: '#1a6640', borderRadius: 100, transition: 'width 1s linear' }} />
            </div>
          </div>

          {/* Live accuracy card */}
          <div style={S.card}>
            <div style={S.sectionLabel}>Live accuracy</div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
              <div style={{ fontFamily: "'Fraunces', serif", fontSize: '2.5rem', fontWeight: 600, letterSpacing: -1, color: result ? (result.is_correct ? '#1a6640' : '#b06a00') : '#c3d9c7' }}>
                {result ? `${Math.round(result.accuracy_score)}%` : '—'}
              </div>
              {result && (
                <span style={{ fontSize: 11, padding: '3px 10px', borderRadius: 100, fontWeight: 500, background: result.is_correct ? '#e8f4ec' : '#fef3e2', color: result.is_correct ? '#1a6640' : '#b06a00' }}>
                  {result.is_correct ? 'Correct' : 'Adjust'}
                </span>
              )}
            </div>

            {/* Confidence bar */}
            {result && (
              <div style={{ marginTop: 10, height: 4, background: '#e8f4ec', borderRadius: 100, overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${result.confidence * 100}%`, background: result.is_correct ? '#1a6640' : '#b06a00', borderRadius: 100, transition: 'width 0.3s ease' }} />
              </div>
            )}
          </div>

          {/* Arm angles card */}
          {result && (
            <div style={S.card}>
              <div style={S.sectionLabel}>Arm angles</div>
              <div style={{ display: 'flex', gap: 10 }}>
                {([['Left', result.left_abduction], ['Right', result.right_abduction]] as [string, number][]).map(([side, angle]) => (
                  <div key={side} style={{ flex: 1, textAlign: 'center', background: '#f7f9f7', borderRadius: 12, padding: '0.75rem 0.5rem' }}>
                    <div style={{ fontSize: 11, color: '#8aaa90', marginBottom: 4 }}>{side}</div>
                    <div style={{ fontFamily: "'Fraunces', serif", fontSize: '1.5rem', fontWeight: 600, color: angle >= 70 ? '#1a6640' : '#b06a00', letterSpacing: -0.5 }}>
                      {Math.round(angle)}°
                    </div>
                    <div style={{ marginTop: 6, height: 4, background: '#e3ede5', borderRadius: 100, overflow: 'hidden' }}>
                      <div style={{ height: '100%', width: `${Math.min((angle / 180) * 100, 100)}%`, background: angle >= 70 ? '#1a6640' : '#b06a00', borderRadius: 100 }} />
                    </div>
                  </div>
                ))}
              </div>
              <p style={{ fontSize: 11, color: '#b5cfb9', textAlign: 'center', marginTop: 8, fontWeight: 300 }}>
                Target: raise arms to 90°
              </p>
            </div>
          )}

          {/* Feedback tip card */}
          {result && (
            <div style={{ ...S.card, background: result.is_correct ? '#f0f9f3' : '#fef8f0', borderColor: result.is_correct ? '#c3d9c7' : '#f5c4a0' }}>
              <div style={S.sectionLabel}>Feedback</div>
              <p style={{ fontSize: 13, color: result.is_correct ? '#1a6640' : '#b06a00', lineHeight: 1.65, fontWeight: 300 }}>
                {result.tip}
              </p>
            </div>
          )}

          {/* Error */}
          {error && (
            <div style={{ background: '#fef3e2', border: '1px solid #f5c4a0', borderRadius: 12, padding: '12px 14px', fontSize: 12, color: '#b06a00' }}>
              {error}
            </div>
          )}

          {/* Start button */}
          {pageStatus === 'ready' && (
            <button
              onClick={startSession}
              style={{ width: '100%', background: '#1a6640', color: '#fff', border: 'none', padding: '14px', borderRadius: 100, fontSize: 15, fontWeight: 500, fontFamily: "'DM Sans', sans-serif", cursor: 'pointer' }}
            >
              ▶ Start session
            </button>
          )}

          {pageStatus === 'loading' && (
            <button
              disabled
              style={{ width: '100%', background: '#e3ede5', color: '#b5cfb9', border: 'none', padding: '14px', borderRadius: 100, fontSize: 15, fontWeight: 500, fontFamily: "'DM Sans', sans-serif", cursor: 'not-allowed' }}
            >
              Loading camera...
            </button>
          )}
        </div>
      </main>
    </div>
  );
}
