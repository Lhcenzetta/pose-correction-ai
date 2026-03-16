'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface User { id: number; first_name: string; last_name: string; email: string; }
interface SessionItem { id: number; exercise_id: number; exercise_name?: string; start_time: string; duration_seconds: number; accuracy_score: number | null; status: string; }

function exerciseIcon(name = '') {
  const n = name.toLowerCase();
  if (n.includes('squat')) return '🏋️';
  if (n.includes('lunge')) return '🦵';
  if (n.includes('shoulder') || n.includes('arm')) return '💪';
  if (n.includes('stretch')) return '🤸';
  return '🏃';
}

function formatDate(iso: string) {
  return new Date(iso).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
}

function formatDuration(secs: number) {
  return secs >= 60 ? `${Math.floor(secs / 60)} min` : `${secs}s`;
}

export default function DashboardPage() {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const token = typeof window !== 'undefined' ? localStorage.getItem('access_token') : null;
  const headers = { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' };

  useEffect(() => {
    if (!token) { router.push('/login'); return; }

    async function load() {
      try {
        const meRes = await fetch(`${API}/me`, { headers });
        if (!meRes.ok) { router.push('/login'); return; }
        const me: User = await meRes.json();
        setUser(me);

        const sessRes = await fetch(`${API}/sessions/user/${me.id}`, { headers });
        const sess = sessRes.ok ? await sessRes.json() : [];
        setSessions(sess);
      } catch {
        setError('Could not load your data. Is your API running?');
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const completed = sessions.filter(s => s.status === 'completed' && s.accuracy_score != null);
  const avgScore = completed.length ? Math.round(completed.reduce((a, s) => a + s.accuracy_score!, 0) / completed.length) : null;
  const bestScore = completed.length ? Math.round(Math.max(...completed.map(s => s.accuracy_score!))) : null;
  const initials = user ? ((user.first_name?.[0] || '') + (user.last_name?.[0] || '')).toUpperCase() : '?';
  const today = new Date().toLocaleDateString('en-GB', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' });
  const sorted = [...sessions].sort((a, b) => new Date(b.start_time).getTime() - new Date(a.start_time).getTime());

  const S = {
    page: { fontFamily: "'DM Sans', sans-serif", background: '#f7f9f7', minHeight: '100vh', color: '#1a2e1f' } as React.CSSProperties,
    nav: { background: '#fff', borderBottom: '1px solid #e3ede5', padding: '1rem 2rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', position: 'sticky' as const, top: 0, zIndex: 10 },
    logo: { fontFamily: "'Fraunces', serif", fontWeight: 600, fontSize: 18, color: '#1a6640' },
    avatar: { width: 34, height: 34, borderRadius: '50%', background: '#e8f4ec', color: '#1a6640', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 13, fontWeight: 500 },
    main: { maxWidth: 900, margin: '0 auto', padding: '2.5rem 2rem' },
    greeting: { fontFamily: "'Fraunces', serif", fontSize: '1.75rem', fontWeight: 600, color: '#0f1f13', letterSpacing: -0.5 },
    statCard: { background: '#fff', border: '1px solid #e3ede5', borderRadius: 16, padding: '1.25rem 1.5rem' },
    statVal: { fontFamily: "'Fraunces', serif", fontSize: '1.75rem', fontWeight: 600, color: '#1a6640', letterSpacing: -1, lineHeight: 1 },
    sessionCard: { background: '#fff', border: '1px solid #e3ede5', borderRadius: 14, padding: '1rem 1.25rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10, cursor: 'pointer' },
    btnStart: { background: '#1a6640', color: '#fff', border: 'none', padding: '11px 24px', borderRadius: 100, fontSize: 13, fontWeight: 500, fontFamily: "'DM Sans', sans-serif", cursor: 'pointer' },
  };

  return (
    <div style={S.page}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,300;0,400;0,600;1,300&family=DM+Sans:wght@300;400;500&display=swap');`}</style>

      <nav style={S.nav}>
        <div style={S.logo}>Pose<span style={{ color: '#b5cfb9' }}>Correct</span></div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={S.avatar}>{initials}</div>
          <span style={{ fontSize: 13, color: '#3a5a42', fontWeight: 500 }}>{user?.first_name} {user?.last_name}</span>
          <button onClick={() => { localStorage.removeItem('access_token'); router.push('/login'); }}
            style={{ fontSize: 12, color: '#8aaa90', background: 'none', border: 'none', cursor: 'pointer', fontFamily: "'DM Sans', sans-serif" }}>
            Sign out
          </button>
        </div>
      </nav>

      <main style={S.main}>
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', marginBottom: '2rem', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <div style={S.greeting}>Good day, <em style={{ color: '#1a6640', fontStyle: 'italic', fontWeight: 300 }}>{user?.first_name || '...'}</em></div>
            <div style={{ fontSize: 12, color: '#8aaa90', marginTop: 4, fontWeight: 300 }}>{today}</div>
          </div>
          <button style={S.btnStart} onClick={() => router.push('/select-exercise')}>▶ Start exercise</button>
        </div>

        {error && <div style={{ background: '#fef3e2', border: '1px solid #f5c4a0', borderRadius: 12, padding: '12px 16px', fontSize: 13, color: '#b06a00', marginBottom: '1.5rem' }}>{error}</div>}

        {/* Stats */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '1rem', marginBottom: '2rem' }}>
          {[
            { label: 'Avg accuracy', value: avgScore != null ? `${avgScore}%` : '—', sub: completed.length ? `${completed.length} sessions` : 'No sessions yet' },
            { label: 'Sessions done', value: sessions.length || '0', sub: `${sessions.length} total` },
            { label: 'Best score', value: bestScore != null ? `${bestScore}%` : '—', sub: 'Personal best' },
          ].map(({ label, value, sub }) => (
            <div key={label} style={S.statCard}>
              <div style={{ fontSize: 11, textTransform: 'uppercase' as const, letterSpacing: '0.07em', color: '#8aaa90', marginBottom: 8 }}>{label}</div>
              <div style={S.statVal}>{value}</div>
              <div style={{ fontSize: 11, color: '#b5cfb9', marginTop: 6, fontWeight: 300 }}>{sub}</div>
            </div>
          ))}
        </div>

        {/* Sessions */}
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '1rem', gap: 12 }}>
          <span style={{ fontSize: 11, textTransform: 'uppercase' as const, letterSpacing: '0.1em', color: '#8aaa90' }}>Previous sessions</span>
          <div style={{ flex: 1, height: 1, background: '#e3ede5' }} />
        </div>

        {loading ? (
          <div style={{ textAlign: 'center', padding: '3rem', color: '#b5cfb9', fontSize: 13 }}>Loading sessions...</div>
        ) : sorted.length === 0 ? (
          <div style={{ background: '#fff', border: '1px dashed #c3d9c7', borderRadius: 16, padding: '3rem 2rem', textAlign: 'center' }}>
            <div style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>🏃</div>
            <div style={{ fontFamily: "'Fraunces', serif", fontSize: '1.1rem', fontWeight: 600, color: '#0f1f13', marginBottom: 8 }}>No sessions yet</div>
            <div style={{ fontSize: 13, color: '#8aaa90', fontWeight: 300, marginBottom: '1.5rem' }}>Complete your first exercise and results will appear here.</div>
            <button onClick={() => router.push('/select-exercise')}
              style={{ background: '#e8f4ec', color: '#1a6640', border: 'none', padding: '10px 22px', borderRadius: 100, fontSize: 13, fontWeight: 500, cursor: 'pointer', fontFamily: "'DM Sans', sans-serif" }}>
              Start your first session
            </button>
          </div>
        ) : (
          sorted.map(s => (
            <div key={s.id} style={S.sessionCard} onClick={() => router.push(`/session/${s.id}`)}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <div style={{ width: 40, height: 40, borderRadius: 12, background: '#e8f4ec', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 18 }}>
                  {exerciseIcon(s.exercise_name)}
                </div>
                <div>
                  <div style={{ fontSize: 14, fontWeight: 500, color: '#0f1f13' }}>{s.exercise_name || `Exercise #${s.exercise_id}`}</div>
                  <div style={{ fontSize: 11, color: '#8aaa90', marginTop: 2, fontWeight: 300 }}>{formatDate(s.start_time)} · {formatDuration(s.duration_seconds)}</div>
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <div style={{ fontFamily: "'Fraunces', serif", fontSize: '1.2rem', fontWeight: 600, color: s.accuracy_score != null && s.accuracy_score < 70 ? '#b06a00' : '#1a6640', letterSpacing: -0.5 }}>
                  {s.accuracy_score != null ? `${Math.round(s.accuracy_score)}%` : '—'}
                </div>
                <span style={{ fontSize: 10, padding: '3px 10px', borderRadius: 100, fontWeight: 500, background: s.status === 'completed' ? '#e8f4ec' : '#fef3e2', color: s.status === 'completed' ? '#1a6640' : '#b06a00' }}>
                  {s.status}
                </span>
              </div>
            </div>
          ))
        )}
      </main>
    </div>
  );
}