'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface Exercise { id: number; name: string; duration_time: number; }

export default function DurationPage() {
  const router = useRouter();
  const [selected, setSelected] = useState<number | null>(5);
  const [custom, setCustom] = useState('');
  const [exercise, setExercise] = useState<Exercise | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const stored = localStorage.getItem('selected_exercise');
    if (!stored) { router.push('/select-exercise'); return; }
    try {
      const ex: Exercise = JSON.parse(stored);
      setExercise(ex);
      if (ex.duration_time && ex.duration_time > 0) setSelected(ex.duration_time);
    } catch (e) {
      router.push('/select-exercise');
    }
  }, []);

  function handlePill(min: number) {
    setSelected(min);
    setCustom('');
  }

  function handleCustom(val: string) {
    setCustom(val);
    const num = parseInt(val, 10);
    setSelected(!isNaN(num) && num > 0 ? num : null);
  }

  async function handleStart() {
    if (!selected || selected <= 0 || !exercise) return;
    setLoading(true);
    setError('');

    const token = localStorage.getItem('access_token');

    try {
      const meRes = await fetch(`${API}/me`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!meRes.ok) throw new Error('Authentication expired. Please log in again.');
      const me = await meRes.json();

      const res = await fetch(`${API}/session`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          user_id: me.id,
          exercise_id: exercise.id,
          duration_seconds: selected * 60,
        }),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || 'Failed to initialize clinical session.');
      }

      const session = await res.json();

      localStorage.setItem('active_session', JSON.stringify({
        session_id: Number(session.id),
        exercise_id: exercise.id,
        exercise_name: exercise.name,
        duration_seconds: selected * 60,
      }));

      router.push(`/session/${session.id}`);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

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
      position: 'sticky' as const,
      top: 0,
      zIndex: 10,
    },
    main: {
      maxWidth: '600px',
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
    pill: (active: boolean) => ({
      background: active ? 'var(--bg-medical)' : 'var(--surface)',
      border: `1px solid ${active ? 'var(--primary)' : 'var(--border)'}`,
      borderRadius: '8px',
      padding: '1rem',
      textAlign: 'center' as const,
      cursor: 'pointer',
      transition: 'all 0.2s',
      boxShadow: active ? '0 0 0 1px var(--primary)' : 'none',
    }),
    btnPrimary: (enabled: boolean) => ({
      background: enabled ? 'var(--primary)' : 'var(--border)',
      color: enabled ? '#fff' : 'var(--text-secondary)',
      border: 'none',
      padding: '14px',
      borderRadius: '6px',
      fontSize: '15px',
      fontWeight: 600,
      cursor: enabled ? 'pointer' : 'not-allowed',
      width: '100%',
      marginTop: '1.5rem',
      transition: 'all 0.2s',
    })
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
        <button onClick={() => router.push('/select-exercise')}
          style={{ fontSize: '13px', color: 'var(--text-secondary)', background: 'none', border: 'none', cursor: 'pointer' }}>
          Back to Selection
        </button>
      </nav>

      <main style={S.main}>
        <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
          <span style={{ fontSize: '10px', fontWeight: 600, color: 'var(--primary)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Step 2 of 2</span>
          <h1 style={{ fontSize: '2rem', fontWeight: 600, marginTop: '0.5rem', marginBottom: '1rem' }}>Set Duration</h1>
          <p style={{ color: 'var(--text-secondary)', margin: '0 auto' }}>
            Configure the recommended time for your <strong>{exercise?.name}</strong> session.
          </p>
        </div>

        <div style={S.card}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', background: 'var(--bg-medical)', border: '1px solid var(--border)', borderRadius: '6px', padding: '12px', fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '2rem' }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0, color: 'var(--primary)' }}>
              <circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>
            </svg>
            <span>Physiotherapists typically recommend 5-10 minutes per session for beginners.</span>
          </div>

          <div style={{ fontSize: '10px', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '12px' }}>Recommended Presets</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '2rem' }}>
            {[3, 5, 10].map(min => (
              <div key={min} style={S.pill(selected === min && custom === '')} onClick={() => handlePill(min)}>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: selected === min && custom === '' ? 'var(--primary)' : 'var(--text-primary)' }}>{min}</div>
                <div style={{ fontSize: '11px', color: 'var(--text-secondary)', fontWeight: 500 }}>MINUTES</div>
              </div>
            ))}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', margin: '2rem 0' }}>
            <div style={{ flex: 1, height: '1px', background: 'var(--border)' }} />
            <span style={{ fontSize: '11px', color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase' }}>Custom Duration</span>
            <div style={{ flex: 1, height: '1px', background: 'var(--border)' }} />
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <input
              type="number" min="1" max="60" placeholder="7"
              value={custom}
              onChange={e => handleCustom(e.target.value)}
              style={{ flex: 1, background: '#fff', border: '1px solid var(--border)', borderRadius: '6px', padding: '12px', fontSize: '1.25rem', fontWeight: 600, color: 'var(--text-primary)', textAlign: 'center', outline: 'none' }}
            />
            <span style={{ fontSize: '14px', color: 'var(--text-secondary)', fontWeight: 500 }}>MINUTES</span>
          </div>

          {error && (
            <div style={{ background: '#FEF2F2', border: '1px solid #FCA5A5', color: '#991B1B', padding: '12px', borderRadius: '6px', fontSize: '13px', marginTop: '1.5rem' }}>
              {error}
            </div>
          )}

          <button
            style={S.btnPrimary(!!selected && selected > 0 && !loading) as any}
            disabled={!selected || selected <= 0 || loading}
            onClick={handleStart}
          >
            {loading ? 'Initializing Session...' : 'Launch Clinical Session'}
          </button>

          <p style={{ fontSize: '11px', color: 'var(--text-secondary)', textAlign: 'center', marginTop: '1rem' }}>
            By starting, you agree to follow the AI guidance safely.
          </p>
        </div>
      </main>
    </div>
  );
}