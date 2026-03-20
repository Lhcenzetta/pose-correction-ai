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
}

interface Feedback {
  id: number;
  session_id: number;
  comment: string;
  score: number;
}

function scoreColor(score: number | null) {
  if (score == null) return 'var(--text-secondary)';
  if (score >= 85) return 'var(--success)';
  if (score >= 70) return 'var(--primary)';
  return '#F59E0B';
}

function scoreLabel(score: number | null) {
  if (score == null) return 'No data';
  if (score >= 90) return 'Exceptional Control';
  if (score >= 75) return 'Strong Performance';
  if (score >= 60) return 'Developing Form';
  return 'Technique Adjustment Recommended';
}

function fmtDuration(secs: number) {
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export default function ResultsPage() {
  const router = useRouter();
  const [session, setSession] = useState<SessionResult | null>(null);
  const [feedback, setFeedback] = useState<Feedback | null>(null);
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
        const meRes = await fetch(`${API}/me`, { headers });
        if (!meRes.ok) throw new Error('Authentication expired');
        const me = await meRes.json();

        const sessRes = await fetch(`${API}/sessions/user/${me.id}`, { headers });
        if (!sessRes.ok) throw new Error('Failed to retrieve clinical data');
        const allSessions: SessionResult[] = await sessRes.json();

        const current = allSessions.find(s => s.id === active.session_id);
        if (!current) throw new Error('Session record not found');
        setSession(current);

        const fbRes = await fetch(`${API}/feedback/${active.session_id}`, { headers });
        if (fbRes.ok) {
          const fbData: Feedback[] = await fbRes.json();
          if (fbData.length > 0) setFeedback(fbData[0]);
        }

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
  const diff = score != null && prevScore != null ? Math.round(score - prevScore) : null;
  const color = scoreColor(score);

  const S = {
    page: {
      backgroundColor: 'var(--bg-medical)',
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column' as const,
    },
    nav: {
      background: 'var(--surface)',
      borderBottom: '1px solid var(--border)',
      padding: '1rem 2rem',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
    },
    main: {
      maxWidth: '800px',
      margin: '0 auto',
      padding: '3rem 2rem',
      width: '100%',
    },
    card: {
      background: 'var(--surface)',
      border: '1px solid var(--border)',
      borderRadius: '8px',
      padding: '2rem',
      boxShadow: 'var(--shadow-subtle)',
    },
    metricBox: {
      textAlign: 'center' as const,
      padding: '1.5rem',
      background: 'var(--bg-medical)',
      borderRadius: '6px',
      border: '1px solid var(--border)',
    },
    label: {
      fontSize: '10px',
      fontWeight: 600,
      color: 'var(--text-secondary)',
      textTransform: 'uppercase' as const,
      letterSpacing: '0.05em',
      marginBottom: '8px',
    },
    btnPrimary: {
      background: 'var(--primary)',
      color: '#fff',
      border: 'none',
      padding: '14px 28px',
      borderRadius: '6px',
      fontSize: '14px',
      fontWeight: 600,
      cursor: 'pointer',
      transition: 'all 0.2s',
    },
    btnSecondary: {
      background: 'var(--surface)',
      color: 'var(--text-primary)',
      border: '1px solid var(--border)',
      padding: '14px 28px',
      borderRadius: '6px',
      fontSize: '14px',
      fontWeight: 600,
      cursor: 'pointer',
      transition: 'all 0.2s',
    }
  };

  if (loading) return (
    <div style={S.page}>
      <div style={{ margin: 'auto', textAlign: 'center', color: 'var(--text-secondary)' }}>
        Processing clinical results...
      </div>
    </div>
  );

  return (
    <div style={S.page}>
      <nav style={S.nav}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--primary)', fontWeight: 700, fontSize: '1.25rem' }}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 4V20M4 12H20" stroke="currentColor" strokeWidth="3" strokeLinecap="round"/>
          </svg>
          <span>PoseCorrect</span>
        </div>
        <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', background: 'var(--bg-medical)', padding: '4px 12px', borderRadius: '100px' }}>
          Final Report: {session?.exercise_name}
        </div>
      </nav>

      <main style={S.main}>
        <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
          <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--success)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '1rem' }}>
            Session Successfully Finalized
          </div>
          <div style={{ fontSize: '6rem', fontWeight: 800, color: color, lineHeight: 1, letterSpacing: '-0.05em', marginBottom: '1rem' }}>
            {score != null ? `${Math.round(score)}%` : '—'}
          </div>
          <h1 style={{ fontSize: '1.5rem', fontWeight: 600, color: 'var(--text-primary)' }}>{scoreLabel(score)}</h1>
          <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
            Protocol: <strong>{session?.exercise_name}</strong> • {new Date(session?.start_time || '').toLocaleDateString()}
          </p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.5rem', marginBottom: '2rem' }}>
          <div style={S.metricBox}>
            <div style={S.label}>Precision</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>{score != null ? `${Math.round(score)}%` : '—'}</div>
          </div>
          <div style={S.metricBox}>
            <div style={S.label}>Duration</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>{fmtDuration(session?.duration_seconds || 0)}</div>
          </div>
          <div style={S.metricBox}>
            <div style={S.label}>Trend</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 700, color: diff && diff >= 0 ? 'var(--success)' : (diff && diff < 0 ? '#EF4444' : 'inherit') }}>
              {diff == null ? 'NEW' : `${diff >= 0 ? '+' : ''}${diff}%`}
            </div>
          </div>
        </div>

        <div style={S.card}>
          <div style={{ ...S.label, marginBottom: '1.5rem' }}>Clinical Analysis & Feedback</div>
          {feedback ? (
            <div style={{ display: 'flex', gap: '1rem', background: 'var(--bg-medical)', padding: '1.5rem', borderRadius: '6px', border: '1px solid var(--border)' }}>
              <div style={{ background: 'rgba(0,119,182,0.1)', color: 'var(--primary)', width: '40px', height: '40px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                </svg>
              </div>
              <div>
                <p style={{ fontSize: '15px', lineHeight: 1.6, color: 'var(--text-primary)', margin: 0 }}>{feedback.comment}</p>
                <div style={{ marginTop: '1rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-secondary)' }}>COACH SCORE:</div>
                  <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--primary)' }}>{feedback.score} / 10</div>
                </div>
              </div>
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: '2rem', background: 'var(--bg-medical)', borderRadius: '6px', border: '1px dashed var(--border)', color: 'var(--text-secondary)' }}>
              AI refinement of session data still in progress.
            </div>
          )}

          <div style={{ marginTop: '2.5rem', paddingTop: '2.5rem', borderTop: '1px solid var(--border)', display: 'flex', gap: '1rem' }}>
            <button 
              style={{ ...S.btnSecondary, flex: 1 }}
              onClick={() => { localStorage.removeItem('active_session'); router.push('/dashboard'); }}
            >
              Return to Dashboard
            </button>
            <button 
              style={{ ...S.btnPrimary, flex: 1 }}
              onClick={() => { localStorage.removeItem('active_session'); router.push('/select-exercise'); }}
            >
              Start New Protocol
            </button>
          </div>
        </div>

        <p style={{ textAlign: 'center', marginTop: '2rem', fontSize: '12px', color: 'var(--text-secondary)' }}>
          This data has been synchronized with your medical profile for further tracking.
        </p>
      </main>
    </div>
  );
}