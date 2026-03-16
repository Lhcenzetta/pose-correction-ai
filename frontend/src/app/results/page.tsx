'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface SessionResult {
  id: number;
  exercise_name: string;
  start_time: string;
  duration_seconds: number;
  accuracy_score: number | null;
  status: string;
  feedback: { comment: string; score: number } | null;
}

function scoreColor(score: number | null) {
  if (score == null) return '#b5cfb9';
  if (score >= 85) return '#1a6640';
  if (score >= 70) return '#3a8a58';
  return '#b06a00';
}

function scoreLabel(score: number | null) {
  if (score == null) return 'No data';
  if (score >= 90) return 'Excellent form!';
  if (score >= 75) return 'Good work, keep it up';
  if (score >= 60) return 'Getting there, keep practising';
  return 'Needs improvement — keep going';
}

function fmtDuration(secs: number) {
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export default function ResultsPage() {
  const router = useRouter();
  const [session, setSession] = useState<SessionResult | null>(null);
  const [prevScore, setPrevScore] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const token = localStorage.getItem('access_token');
    if (!token) { router.push('/login'); return; }

    const activeRaw = localStorage.getItem('active_session');
    if (!activeRaw) { router.push('/dashboard'); return; }
    const active = JSON.parse(activeRaw);
    const headers = { Authorization: `Bearer ${token}` };

    async function load() {
      try {
        // Load current session result via dashboard endpoint
        const meRes = await fetch(`${API}/me`, { headers });
        if (!meRes.ok) throw new Error('Auth failed');
        const me = await meRes.json();

        const sessRes = await fetch(`${API}/sessions/user/${me.id}`, { headers });
        if (!sessRes.ok) throw new Error('Could not load sessions');
        const allSessions: SessionResult[] = await sessRes.json();

        const current = allSessions.find(s => s.id === active.session_id);
        if (!current) throw new Error('Session not found');
        setSession(current);

        // Find previous session for comparison
        const others = allSessions
          .filter(s => s.id !== active.session_id && s.accuracy_score != null)
          .sort((a, b) => new Date(b.start_time).getTime() - new Date(a.start_time).getTime());
        if (others.length > 0) setPrevScore(others[0].accuracy_score!);
      } catch (e: any) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const score = session?.accuracy_score ?? null;
  const diff  = score != null && prevScore != null ? Math.round(score - prevScore) : null;
  const color = scoreColor(score);

  const S: Record<string, React.CSSProperties> = {
    page:   { fontFamily: "'DM Sans', sans-serif", background: '#f7f9f7', minHeight: '100vh' },
    nav:    { background: '#fff', borderBottom: '1px solid #e3ede5', padding: '1rem 2rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', position: 'sticky', top: 0, zIndex: 10 },
    main:   { maxWidth: 720, margin: '0 auto', padding: '3rem 2rem' },
    card:   { background: '#fff', border: '1px solid #e3ede5', borderRadius: 16, padding: '1.25rem 1.5rem' },
    label:  { fontSize: 11, textTransform: 'uppercase' as const, letterSpacing: '0.08em', color: '#8aaa90', marginBottom: 6 },
    statVal:{ fontFamily: "'Fraunces', serif", fontWeight: 600, letterSpacing: -1, lineHeight: 1 },
  };

  if (loading) return (
    <div style={S.page}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,300;0,400;0,600;1,300&family=DM+Sans:wght@300;400;500&display=swap');`}</style>
      <nav style={S.nav}><div style={{ fontFamily: "'Fraunces', serif", fontWeight: 600, fontSize: 18, color: '#1a6640' }}>Pose<span style={{ color: '#b5cfb9' }}>Correct</span></div></nav>
      <div style={{ textAlign: 'center', padding: '5rem', color: '#b5cfb9', fontSize: 13 }}>Loading results...</div>
    </div>
  );

  if (error) return (
    <div style={S.page}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,300;0,400;0,600;1,300&family=DM+Sans:wght@300;400;500&display=swap');`}</style>
      <nav style={S.nav}><div style={{ fontFamily: "'Fraunces', serif", fontWeight: 600, fontSize: 18, color: '#1a6640' }}>Pose<span style={{ color: '#b5cfb9' }}>Correct</span></div></nav>
      <div style={{ textAlign: 'center', padding: '5rem' }}>
        <div style={{ color: '#b06a00', fontSize: 13, marginBottom: '1rem' }}>{error}</div>
        <button onClick={() => router.push('/dashboard')} style={{ background: '#1a6640', color: '#fff', border: 'none', padding: '10px 24px', borderRadius: 100, fontSize: 13, cursor: 'pointer', fontFamily: "'DM Sans', sans-serif" }}>Go to dashboard</button>
      </div>
    </div>
  );

  return (
    <div style={S.page}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,300;0,400;0,600;1,300&family=DM+Sans:wght@300;400;500&display=swap');`}</style>

      <nav style={S.nav}>
        <div style={{ fontFamily: "'Fraunces', serif", fontWeight: 600, fontSize: 18, color: '#1a6640' }}>
          Pose<span style={{ color: '#b5cfb9' }}>Correct</span>
        </div>
        <span style={{ fontSize: 12, color: '#8aaa90' }}>{session?.exercise_name}</span>
      </nav>

      <main style={S.main}>

        {/* ── Hero score ── */}
        <div style={{ textAlign: 'center', marginBottom: '2.5rem' }}>
          <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.1em', color: '#1a6640', marginBottom: '1rem' }}>
            ✦ Session complete ✦
          </div>
          <div style={{ fontFamily: "'Fraunces', serif", fontSize: 'clamp(5rem, 15vw, 8rem)', fontWeight: 600, color: color, letterSpacing: -4, lineHeight: 1, marginBottom: '0.75rem' }}>
            {score != null ? `${Math.round(score)}%` : '—'}
          </div>
          <div style={{ fontSize: 14, color: '#8aaa90', fontWeight: 300 }}>
            {scoreLabel(score)} — <span style={{ color: '#3a5a42', fontWeight: 500 }}>{session?.exercise_name}</span>
          </div>
        </div>

        {/* ── Stats grid ── */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2,1fr)', gap: '1rem', marginBottom: '1.5rem' }}>

          <div style={S.card}>
            <div style={S.label}>Accuracy score</div>
            <div style={{ ...S.statVal, fontSize: '2rem', color }}>{score != null ? `${Math.round(score)}%` : '—'}</div>
            <div style={{ fontSize: 11, color: '#b5cfb9', marginTop: 6, fontWeight: 300 }}>Overall session score</div>
          </div>

          <div style={S.card}>
            <div style={S.label}>Duration</div>
            <div style={{ ...S.statVal, fontSize: '2rem', color: '#0f1f13' }}>{fmtDuration(session?.duration_seconds ?? 0)}</div>
            <div style={{ fontSize: 11, color: '#b5cfb9', marginTop: 6, fontWeight: 300 }}>Full session completed ✓</div>
          </div>

          <div style={S.card}>
            <div style={S.label}>vs last session</div>
            <div style={{ ...S.statVal, fontSize: '2rem', color: diff == null ? '#b5cfb9' : diff >= 0 ? '#1a6640' : '#b06a00' }}>
              {diff == null ? '—' : `${diff >= 0 ? '+' : ''}${diff}%`}
            </div>
            <div style={{ fontSize: 11, color: '#b5cfb9', marginTop: 6, fontWeight: 300 }}>
              {diff == null ? 'First session' : diff >= 0 ? 'Great improvement!' : 'Keep going!'}
            </div>
          </div>

          <div style={S.card}>
            <div style={S.label}>Status</div>
            <div style={{ ...S.statVal, fontSize: '2rem', color: session?.status === 'completed' ? '#1a6640' : '#b06a00' }}>
              {session?.status === 'completed' ? 'Done' : 'Pending'}
            </div>
            <div style={{ fontSize: 11, color: '#b5cfb9', marginTop: 6, fontWeight: 300 }}>{new Date(session?.start_time ?? '').toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })}</div>
          </div>
        </div>

        {/* ── Accuracy bar ── */}
        <div style={{ ...S.card, marginBottom: '1.5rem' }}>
          <div style={S.label}>Score distribution</div>
          <div style={{ marginBottom: '0.75rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#5a7a62', marginBottom: 6 }}>
              <span>Accuracy</span><span>{score != null ? `${Math.round(score)}%` : '—'}</span>
            </div>
            <div style={{ height: 10, background: '#e8f4ec', borderRadius: 100, overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${score ?? 0}%`, background: color, borderRadius: 100, transition: 'width 1s ease' }} />
            </div>
          </div>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#8aaa90', marginBottom: 6 }}>
              <span>Remaining</span><span>{score != null ? `${Math.round(100 - score)}%` : '—'}</span>
            </div>
            <div style={{ height: 10, background: '#e8f4ec', borderRadius: 100, overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${score != null ? 100 - score : 0}%`, background: '#f5c4a0', borderRadius: 100, transition: 'width 1s ease' }} />
            </div>
          </div>
        </div>

        {/* ── AI Feedback ── */}
        {session?.feedback && (
          <div style={{ ...S.card, marginBottom: '1.5rem', background: score != null && score >= 75 ? '#f0f9f3' : '#fef8f0', borderColor: score != null && score >= 75 ? '#c3d9c7' : '#f5c4a0' }}>
            <div style={S.label}>💡 AI feedback</div>
            <p style={{ fontSize: 14, color, lineHeight: 1.7, fontWeight: 300 }}>{session.feedback.comment}</p>
          </div>
        )}

        {/* ── Actions ── */}
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          <button
            onClick={() => {
              localStorage.removeItem('active_session');
              router.push('/duration');
            }}
            style={{ flex: 1, minWidth: 140, background: '#f7f9f7', color: '#3a5a42', border: '1px solid #c3d9c7', padding: '13px', borderRadius: 100, fontSize: 14, fontWeight: 500, cursor: 'pointer', fontFamily: "'DM Sans', sans-serif" }}
          >
            ↩ Retry
          </button>
          <button
            onClick={() => {
              localStorage.removeItem('active_session');
              router.push('/dashboard');
            }}
            style={{ flex: 1, minWidth: 140, background: '#1a6640', color: '#fff', border: 'none', padding: '13px', borderRadius: 100, fontSize: 14, fontWeight: 500, cursor: 'pointer', fontFamily: "'DM Sans', sans-serif" }}
          >
            Dashboard →
          </button>
        </div>
      </main>
    </div>
  );
}