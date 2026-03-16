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
    const ex: Exercise = JSON.parse(stored);
    setExercise(ex);
    if (ex.duration_time) setSelected(ex.duration_time);
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
      // Step 1 — get real user id
      const meRes = await fetch(`${API}/me`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!meRes.ok) throw new Error('Auth failed — please log in again.');
      const me = await meRes.json();

      // Step 2 — create session
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
        throw new Error(data.detail || 'Failed to create session.');
      }

      const session = await res.json();

      localStorage.setItem('active_session', JSON.stringify({
        session_id: Number(session.id),
        exercise_id: exercise.id,
        exercise_name: exercise.name,
        duration_seconds: selected * 60,
      }));

      router.push('/session');
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const S = {
    page: { fontFamily: "'DM Sans', sans-serif", background: '#f7f9f7', minHeight: '100vh' } as React.CSSProperties,
    nav: { background: '#fff', borderBottom: '1px solid #e3ede5', padding: '1rem 2rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', position: 'sticky' as const, top: 0, zIndex: 10 },
    main: { maxWidth: 520, margin: '0 auto', padding: '3rem 2rem', display: 'flex', flexDirection: 'column' as const, alignItems: 'center' },
    card: { background: '#fff', border: '1px solid #e3ede5', borderRadius: 20, padding: '2rem', width: '100%' },
    pill: (active: boolean): React.CSSProperties => ({
      background: active ? '#f0f9f3' : '#f7f9f7',
      border: `1.5px solid ${active ? '#1a6640' : '#e3ede5'}`,
      borderRadius: 14, padding: '1rem 0.75rem',
      textAlign: 'center', cursor: 'pointer', transition: 'all 0.2s',
      boxShadow: active ? '0 0 0 3px rgba(26,102,64,0.08)' : 'none',
    }),
    btnStart: (enabled: boolean): React.CSSProperties => ({
      width: '100%', background: enabled ? '#1a6640' : '#e3ede5',
      color: enabled ? '#fff' : '#b5cfb9', border: 'none',
      padding: 14, borderRadius: 100, fontSize: 15, fontWeight: 500,
      fontFamily: "'DM Sans', sans-serif",
      cursor: enabled ? 'pointer' : 'not-allowed', transition: 'all 0.2s',
    }),
  };

  return (
    <div style={S.page}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,300;0,400;0,600;1,300&family=DM+Sans:wght@300;400;500&display=swap');`}</style>

      <nav style={S.nav}>
        <div style={{ fontFamily: "'Fraunces', serif", fontWeight: 600, fontSize: 18, color: '#1a6640' }}>
          Pose<span style={{ color: '#b5cfb9' }}>Correct</span>
        </div>
        <button onClick={() => router.push('/select-exercise')}
          style={{ fontSize: 12, color: '#8aaa90', background: 'none', border: 'none', cursor: 'pointer', fontFamily: "'DM Sans', sans-serif" }}>
          ← Back
        </button>
      </nav>

      <main style={S.main}>
        <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.1em', color: '#1a6640', marginBottom: 6 }}>Step 2 of 2</div>
        <h1 style={{ fontFamily: "'Fraunces', serif", fontSize: '2rem', fontWeight: 600, color: '#0f1f13', letterSpacing: -1, textAlign: 'center', marginBottom: 6 }}>Select duration</h1>
        <p style={{ fontSize: 13, color: '#8aaa90', fontWeight: 300, textAlign: 'center', marginBottom: '2rem' }}>Follow your doctor's recommendation</p>

        <div style={S.card}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, background: '#f0f9f3', border: '1px solid #c3d9c7', borderRadius: 12, padding: '12px 16px', fontSize: 13, color: '#3a5a42', marginBottom: '1.75rem' }}>
            <span>ℹ️</span>
            <span>Please enter the session duration recommended by your doctor.</span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 10, marginBottom: '1.5rem' }}>
            {[2, 5, 10].map(min => (
              <div key={min} style={S.pill(selected === min && custom === '')} onClick={() => handlePill(min)}>
                <div style={{ fontFamily: "'Fraunces', serif", fontSize: '1.75rem', fontWeight: 600, color: '#1a6640', letterSpacing: -1, lineHeight: 1 }}>{min}</div>
                <div style={{ fontSize: 11, color: '#8aaa90', marginTop: 4, textTransform: 'uppercase', letterSpacing: '0.05em' }}>min</div>
              </div>
            ))}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 12, margin: '1.25rem 0' }}>
            <div style={{ flex: 1, height: 1, background: '#e3ede5' }} />
            <span style={{ fontSize: 11, color: '#b5cfb9', textTransform: 'uppercase', letterSpacing: '0.08em' }}>or custom</span>
            <div style={{ flex: 1, height: 1, background: '#e3ede5' }} />
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: '1.75rem' }}>
            <input
              type="number" min="1" max="60" placeholder="e.g. 7"
              value={custom}
              onChange={e => handleCustom(e.target.value)}
              style={{ flex: 1, background: '#f7f9f7', border: '1.5px solid #e3ede5', borderRadius: 12, padding: '12px 16px', fontSize: '1.25rem', fontWeight: 500, fontFamily: "'Fraunces', serif", color: '#0f1f13', textAlign: 'center', outline: 'none' }}
            />
            <span style={{ fontSize: 13, color: '#8aaa90', fontWeight: 300 }}>minutes</span>
          </div>

          {error && (
            <div style={{ background: '#fef3e2', border: '1px solid #f5c4a0', borderRadius: 10, padding: '10px 14px', fontSize: 12, color: '#b06a00', marginBottom: '1rem' }}>
              {error}
            </div>
          )}

          <button
            style={S.btnStart(!!selected && selected > 0 && !loading)}
            disabled={!selected || selected <= 0 || loading}
            onClick={handleStart}
          >
            {loading ? 'Creating session...' : 'Start session →'}
          </button>

          <p style={{ fontSize: 12, color: '#b5cfb9', textAlign: 'center', marginTop: 10, fontWeight: 300 }}>
            {selected && selected > 0 ? `${selected} minute${selected > 1 ? 's' : ''} selected` : 'Choose a duration to continue'}
          </p>
        </div>
      </main>
    </div>
  );
}