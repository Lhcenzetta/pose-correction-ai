'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import PoseGuide from '@/components/PoseGuide';

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
  const [hasCompletedGuide, setHasCompletedGuide] = useState(false);

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

  const finalizeSession = useCallback(async (sid: number) => {
    const token = localStorage.getItem('access_token');
    try {
      await fetch(`${API}/session/${sid}/finalize`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      });
    } catch { /* session already saved in DB */ }

    if (timerRef.current) clearInterval(timerRef.current);
    if (cameraRef.current) cameraRef.current.stop();
    router.push('/results');
  }, [router]);

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
      }
    } catch (e) {
      console.error('sendPose failed:', e);
    } finally {
      isSendingRef.current = false;
    }
  }, []);

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
    upperBodyIds.forEach(id => {
      const lm = landmarks[id];
      features.push(lm.x, lm.y, lm.z);
    });
    features.push(
      calcAngle(landmarks[11], landmarks[13], landmarks[15]),
      calcAngle(landmarks[12], landmarks[14], landmarks[16]),
      calcAngle(landmarks[23], landmarks[11], landmarks[13]),
      calcAngle(landmarks[24], landmarks[12], landmarks[14]),
    );
    return features;
  }

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
          color: 'rgba(0,119,182,0.6)',
          lineWidth: 2,
        });
      }
      if (w.drawLandmarks) {
        w.drawLandmarks(ctx, results.poseLandmarks, {
          color: '#00B4D8',
          lineWidth: 1,
          radius: 3,
        });
      }
      if (pageStatus === 'running') {
        const features = extractFeatures(results.poseLandmarks);
        sendPose(features);
      }
    }
    ctx.restore();
  }, [pageStatus, sendPose]);

  useEffect(() => {
    const raw = localStorage.getItem('active_session');
    if (!raw) { router.push('/select-exercise'); return; }
    const parsed: SessionConfig = JSON.parse(raw);
    setConfig(parsed);
    sessionRef.current = parsed;
    setTimeLeft(parsed.duration_seconds);

    async function init() {
      try {
        await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/pose/pose.js');
        await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js');
        await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js');
        const w = window as any;
        const pose = new w.Pose({
          locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
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
          onFrame: async () => { await pose.send({ image: videoRef.current }); },
          width: 640,
          height: 480,
        });
        cameraRef.current = camera;
        await camera.start();
        setPageStatus('ready');
      } catch (e) {
        setError('Camera initialization failed. Please verify permissions.');
      }
    }
    init();
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (cameraRef.current) cameraRef.current.stop();
    };
  }, []);

  useEffect(() => {
    if (poseRef.current) poseRef.current.onResults(onResults);
  }, [onResults]);

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

  function fmtTime(s: number) {
    return `${Math.floor(s / 60).toString().padStart(2, '0')}:${(s % 60).toString().padStart(2, '0')}`;
  }

  const S = {
    page: {
      backgroundColor: 'var(--bg-medical)',
      minHeight: '100vh',
      color: 'var(--text-primary)',
    },
    nav: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0.75rem 2rem',
      background: 'var(--surface)',
      borderBottom: '1px solid var(--border)',
      position: 'sticky' as const,
      top: 0,
      zIndex: 10,
    },
    logo: {
      fontWeight: 600,
      fontSize: '1.25rem',
      color: 'var(--primary)',
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      cursor: 'pointer',
    },
    main: {
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '2rem',
      display: 'grid',
      gridTemplateColumns: '1fr 340px',
      gap: '2rem',
    },
    camWrap: {
      background: '#000',
      borderRadius: '12px',
      overflow: 'hidden',
      position: 'relative' as const,
      aspectRatio: '16/9',
      boxShadow: '0 8px 30px rgba(0,0,0,0.2)',
      border: '1px solid var(--border)',
    },
    sidebar: {
      display: 'flex',
      flexDirection: 'column' as const,
      gap: '1.25rem',
    },
    card: {
      background: 'var(--surface)',
      border: '1px solid var(--border)',
      borderRadius: '8px',
      padding: '1.5rem',
      boxShadow: 'var(--shadow-subtle)',
    },
    metricLabel: {
      fontSize: '10px',
      fontWeight: 600,
      textTransform: 'uppercase' as const,
      letterSpacing: '0.05em',
      color: 'var(--text-secondary)',
      marginBottom: '8px',
    },
    metricValue: {
      fontSize: '2rem',
      fontWeight: 700,
      color: 'var(--text-primary)',
    },
    btnPrimary: {
      background: 'var(--primary)',
      color: '#fff',
      border: 'none',
      padding: '14px',
      borderRadius: '6px',
      fontSize: '15px',
      fontWeight: 600,
      cursor: 'pointer',
      width: '100%',
      transition: 'opacity 0.2s',
    }
  };

  const CrossIcon = () => (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 4V20M4 12H20" stroke="currentColor" strokeWidth="3" strokeLinecap="round"/>
    </svg>
  );

  return (
    <>
      <div style={S.page}>
        <style>{`
          @keyframes flash { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
          .live-indicator { animation: flash 1.5s infinite; }
        `}</style>
        
        <nav style={S.nav}>
          <div style={S.logo} onClick={() => router.push('/')}>
            <CrossIcon />
            <span>PoseCorrect</span>
            <div style={{ width: '1px', height: '20px', background: 'var(--border)', margin: '0 12px' }} />
            <div style={{ fontSize: '14px', fontWeight: 500, color: 'var(--text-secondary)' }}>{config?.exercise_name}</div>
          </div>
          {pageStatus === 'running' && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: 'rgba(239,68,68,0.08)', padding: '6px 14px', borderRadius: '20px', border: '1px solid rgba(239,68,68,0.2)' }}>
              <div className="live-indicator" style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#EF4444' }} />
              <span style={{ fontSize: '11px', fontWeight: 700, color: '#EF4444', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Session Live Analysis</span>
            </div>
          )}
        </nav>

        <main style={S.main}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <div style={S.camWrap}>
              <video ref={videoRef} style={{ display: 'none' }} />
              <canvas ref={canvasRef} width={640} height={480} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
              
              {pageStatus !== 'running' && (
                <div style={{ position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', color: '#fff' }}>
                  {pageStatus === 'loading' ? 'CALIBRATING SENSOR...' : 'SENSOR READY'}
                </div>
              )}

              {pageStatus === 'running' && result && (
                <div style={{ position: 'absolute', top: '20px', left: '20px', background: 'rgba(0,0,0,0.7)', padding: '10px 16px', borderRadius: '4px', borderLeft: `4px solid ${result.is_correct ? 'var(--success)' : '#EF4444'}` }}>
                  <div style={{ fontSize: '10px', color: '#94A3B8', textTransform: 'uppercase', fontWeight: 600 }}>Clinical Advice</div>
                  <div style={{ color: '#fff', fontSize: '14px', fontWeight: 500 }}>{result.tip}</div>
                </div>
              )}
            </div>

            {pageStatus === 'running' && (
              <div style={{ ...S.card, display: 'flex', alignItems: 'center', gap: '2rem' }}>
                <div style={{ flex: 1 }}>
                  <div style={S.metricLabel}>Session Progress</div>
                  <div style={{ height: '8px', background: 'var(--bg-medical)', borderRadius: '4px', overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${((config!.duration_seconds - timeLeft) / config!.duration_seconds) * 100}%`, background: 'var(--primary)', transition: 'width 1s linear' }} />
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={S.metricLabel}>Time Elapsed</div>
                  <div style={{ fontSize: '1.25rem', fontWeight: 700 }}>{fmtTime(config!.duration_seconds - timeLeft)} / {fmtTime(config!.duration_seconds)}</div>
                </div>
              </div>
            )}
          </div>

          <aside style={S.sidebar}>
            <div style={S.card}>
              <div style={S.metricLabel}>Correction Score</div>
              
              {/* Conditional Score Line */}
              <div style={{ 
                height: '4px', 
                width: '100%', 
                background: 'var(--border)', 
                borderRadius: '2px', 
                margin: '4px 0 12px 0', 
                overflow: 'hidden' 
              }}>
                <div style={{ 
                  height: '100%', 
                  width: result ? `${Math.min(100, Math.max(0, result.accuracy_score))}%` : '0%', 
                  background: result ? (result.accuracy_score > 50 ? 'var(--success)' : '#EF4444') : 'var(--border)',
                  transition: 'width 0.4s cubic-bezier(0.4, 0, 0.2, 1), background-color 0.3s ease'
                }} />
              </div>

              <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
                <div style={S.metricValue}>{result ? `${Math.round(result.accuracy_score)}%` : '—'}</div>
                <div style={{ fontSize: '12px', fontWeight: 600, color: result?.is_correct ? 'var(--success)' : '#EF4444' }}>
                  {result?.is_correct ? 'OPTIMAL' : 'ADJUST'}
                </div>
              </div>
            </div>

            <div style={S.card}>
              <div style={S.metricLabel}>Time Remaining</div>
              <div style={{ ...S.metricValue, color: timeLeft <= 10 && pageStatus === 'running' ? '#EF4444' : 'var(--text-primary)' }}>
                {fmtTime(timeLeft)}
              </div>
            </div>

            {result && (
              <div style={S.card}>
                <div style={S.metricLabel}>Clinical Metrics</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Left Abduction</span>
                    <span style={{ fontSize: '14px', fontWeight: 700 }}>{Math.round(result.left_abduction)}°</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Right Abduction</span>
                    <span style={{ fontSize: '14px', fontWeight: 700 }}>{Math.round(result.right_abduction)}°</span>
                  </div>
                </div>
              </div>
            )}

            <div style={{ ...S.card, background: 'var(--bg-medical)', border: '1px dashed var(--border)' }}>
              <div style={S.metricLabel}>Sensor Status</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: pageStatus !== 'loading' ? 'var(--success)' : '#F59E0B' }} />
                <span style={{ fontSize: '12px', fontWeight: 600 }}>{pageStatus === 'loading' ? 'SYNCING...' : 'ONLINE'}</span>
              </div>
            </div>

            {pageStatus === 'ready' && (
              <button 
                onClick={startSession}
                style={S.btnPrimary}
                onMouseOver={(e) => e.currentTarget.style.opacity = '0.9'}
                onMouseOut={(e) => e.currentTarget.style.opacity = '1'}
              >
                Launch Protocol
              </button>
            )}
          </aside>
        </main>
      </div>

      {!hasCompletedGuide && (
        <div style={{ position: 'fixed', inset: 0, zIndex: 100, overflow: 'auto' }}>
          <PoseGuide
            exerciseName={config?.exercise_name || 'Protocol'}
            poseImageSrc={[
              "/Gemini_Generated_Image_cdr8a4cdr8a4cdr8.png",
              "/Gemini_Generated_Image_l281ndl281ndl281.png"
            ]}
            instructions={[
              "Orient your device so your full upper body is visible.",
              "Adopt a neutral standing posture with shoulders relaxed.",
              "Follow the on-screen tips for precise joint alignment.",
              "Maintain steady controlled movements throughout."
            ]}
            onStart={() => { setHasCompletedGuide(true); startSession(); }}
            onBack={() => router.push('/select-exercise')}
          />
        </div>
      )}
    </>
  );
}