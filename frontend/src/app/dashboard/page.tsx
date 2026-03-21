'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface User { id: number; first_name: string; last_name: string; email: string; }
interface SessionItem { id: number; exercise_id: number; exercise_name?: string; start_time: string; duration_seconds: number; accuracy_score: number | null; status: string; }

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
        setError('Could not load medical data. Please refresh.');
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
  const today = new Date().toLocaleDateString('en-GB', { weekday: 'long', day: 'numeric', month: 'long' });
  const sorted = [...sessions].sort((a, b) => new Date(b.start_time).getTime() - new Date(a.start_time).getTime());

  const S = {
    page: {
      backgroundColor: 'var(--bg-medical)',
      minHeight: '100vh',
    },
    nav: {
      background: 'var(--surface)',
      borderBottom: '1px solid var(--border)',
      padding: '0.75rem 2rem',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      position: 'sticky' as const,
      top: 0,
      zIndex: 10,
    },
    main: {
      maxWidth: '1000px',
      margin: '0 auto',
      padding: '2rem',
    },
    card: {
      background: 'var(--surface)',
      border: '1px solid var(--border)',
      borderRadius: '8px',
      padding: '1.5rem',
      boxShadow: 'var(--shadow-subtle)',
    },
    statLabel: {
      fontSize: '10px',
      fontWeight: 600,
      textTransform: 'uppercase' as const,
      letterSpacing: '0.05em',
      color: 'var(--text-secondary)',
      marginBottom: '8px',
    },
    statValue: {
      fontSize: '1.5rem',
      fontWeight: 700,
      color: 'var(--text-primary)',
    },
    btnPrimary: {
      background: 'var(--primary)',
      color: '#fff',
      border: 'none',
      padding: '10px 20px',
      borderRadius: '6px',
      fontSize: '14px',
      fontWeight: 500,
      cursor: 'pointer',
    }
  };

  return (
    <div style={S.page}>
      <nav style={S.nav}>
        <div 
          style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--primary)', fontWeight: 700, fontSize: '1.25rem', cursor: 'pointer' }}
          onClick={() => router.push('/')}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 4V20M4 12H20" stroke="currentColor" strokeWidth="3" strokeLinecap="round"/>
          </svg>
          <span>PoseCorrect</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '14px', fontWeight: 600 }}>{user?.first_name} {user?.last_name}</div>
            <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Patient ID: #{user?.id}</div>
          </div>
          <div style={{ width: '36px', height: '36px', borderRadius: '50%', background: 'var(--bg-medical)', color: 'var(--primary)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, fontSize: '14px', border: '1px solid var(--border)' }}>
            {initials}
          </div>
          <button 
            onClick={() => { localStorage.removeItem('access_token'); router.push('/login'); }}
            style={{ padding: '6px', background: 'none', border: 'none', color: '#EF4444', cursor: 'pointer' }}
            title="Sign Out"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" /><polyline points="16 17 21 12 16 7" /><line x1="21" y1="12" x2="9" y2="12" />
            </svg>
          </button>
        </div>
      </nav>

      <main style={S.main}>
        <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
          <div>
            <h1 style={{ fontSize: '1.75rem', fontWeight: 700 }}>Patient Dashboard</h1>
            <p style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>Welcome back. Today is {today}.</p>
          </div>
          <button style={S.btnPrimary} onClick={() => router.push('/select-exercise')}>
            Start New Session
          </button>
        </header>

        {error && (
          <div style={{ background: '#FEF2F2', border: '1px solid #FCA5A5', color: '#991B1B', padding: '12px', borderRadius: '6px', fontSize: '13px', marginBottom: '2rem' }}>
            {error}
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.5rem', marginBottom: '2.5rem' }}>
          <div style={S.card}>
            <div style={S.statLabel}>Avg Recovery Score</div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
              <div style={S.statValue}>{avgScore != null ? `${avgScore}%` : '—'}</div>
              <div style={{ fontSize: '11px', color: 'var(--success)', fontWeight: 600 }}>Optimal</div>
            </div>
            <div style={{ height: '4px', background: 'var(--bg-medical)', borderRadius: '2px', marginTop: '12px' }}>
              <div style={{ height: '100%', width: avgScore ? `${avgScore}%` : '0%', background: 'var(--primary)', borderRadius: '2px' }} />
            </div>
          </div>
          <div style={S.card}>
            <div style={S.statLabel}>Total Sessions</div>
            <div style={S.statValue}>{sessions.length || '0'}</div>
            <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '8px' }}>Completed programs</div>
          </div>
          <div style={S.card}>
            <div style={S.statLabel}>Peak Performance</div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
              <div style={S.statValue}>{bestScore != null ? `${bestScore}%` : '—'}</div>
              <div style={{ fontSize: '11px', color: 'var(--primary)', fontWeight: 600 }}>Personal Best</div>
            </div>
          </div>
        </div>

        <section style={S.card}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
            <h2 style={{ fontSize: '1.1rem', fontWeight: 600 }}>Recent Activity</h2>
            <Link href="#" style={{ fontSize: '12px', color: 'var(--primary)', textDecoration: 'none', fontWeight: 500 }}>View All</Link>
          </div>

          {loading ? (
            <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-secondary)' }}>Loading transactions...</div>
          ) : sorted.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '3rem 0', color: 'var(--text-secondary)' }}>
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" style={{ marginBottom: '1rem', opacity: 0.5 }}>
                <rect x="3" y="4" width="18" height="18" rx="2" ry="2" /><line x1="16" y1="2" x2="16" y2="6" /><line x1="8" y1="2" x2="8" y2="6" /><line x1="3" y1="10" x2="21" y2="10" />
              </svg>
              <p>No exercise history found. Start your first session to track progress.</p>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1px', background: 'var(--border)' }}>
              {sorted.map(s => (
                <div key={s.id} style={{ background: 'var(--surface)', padding: '1rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <div style={{ background: 'var(--bg-medical)', color: 'var(--primary)', padding: '8px', borderRadius: '6px' }}>
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M18 15V18H6V15" /><circle cx="12" cy="7" r="4" />
                      </svg>
                    </div>
                    <div>
                      <div style={{ fontSize: '14px', fontWeight: 600 }}>{s.exercise_name || `Session #${s.id}`}</div>
                      <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>{formatDate(s.start_time)} · {formatDuration(s.duration_seconds)}</div>
                    </div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '2rem' }}>
                    <div style={{ textAlign: 'right' }}>
                      <div style={{ fontSize: '14px', fontWeight: 700, color: s.accuracy_score != null && s.accuracy_score >= 80 ? 'var(--success)' : 'var(--text-primary)' }}>
                        {s.accuracy_score != null ? `${Math.round(s.accuracy_score)}%` : '—'}
                      </div>
                      <div style={{ fontSize: '10px', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>Accuracy</div>
                    </div>
                    <div style={{ 
                      fontSize: '10px', 
                      fontWeight: 600, 
                      padding: '4px 8px', 
                      borderRadius: '4px', 
                      background: s.status === 'completed' ? '#DCFCE7' : '#FEF3C7',
                      color: s.status === 'completed' ? '#166534' : '#92400E'
                    }}>
                      {s.status.toUpperCase()}
                    </div>
                    <button 
                      onClick={() => router.push(`/session/${s.id}`)}
                      style={{ background: 'none', border: 'none', color: 'var(--primary)', cursor: 'pointer' }}
                    >
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="9 18 15 12 9 6" />
                      </svg>
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}