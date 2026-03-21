'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface Exercise {
  id: number;
  name: string;
  description: string;
  duration_time: number;
}

const SOON: Exercise[] = [
  { id: -1, name: 'Squats', description: 'Lower your hips from a standing position to strengthen legs and core.', duration_time: 0 },
];

export default function SelectExercisePage() {
  const router = useRouter();
  const [exercises, setExercises] = useState<Exercise[]>([]);
  const [selected, setSelected] = useState<Exercise | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const token = localStorage.getItem('access_token');
    if (!token) {
      router.push('/login');
      return;
    }

    fetch(`${API}/exercises/`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(data => {
        const list = Array.isArray(data) ? data : [];
        setExercises(list);
      })
      .catch(() => {
        setError('Could not load exercises. Please try again later.');
        setExercises([]);
      })
      .finally(() => setLoading(false));
  }, []);

  function handleContinue() {
    if (!selected) return;
    localStorage.setItem('selected_exercise', JSON.stringify(selected));
    router.push('/duration');
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
      maxWidth: '800px',
      margin: '0 auto',
      padding: '3rem 2rem',
      width: '100%',
    },
    grid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))',
      gap: '1.5rem',
      marginBottom: '3rem',
    },
    card: (isSelected: boolean, disabled: boolean) => ({
      background: 'var(--surface)',
      border: `1px solid ${isSelected ? 'var(--primary)' : 'var(--border)'}`,
      borderRadius: '8px',
      padding: '1.5rem',
      cursor: disabled ? 'not-allowed' : 'pointer',
      opacity: disabled ? 0.6 : 1,
      transition: 'all 0.2s',
      position: 'relative' as const,
      display: 'flex',
      flexDirection: 'column' as const,
      boxShadow: isSelected ? '0 0 0 1px var(--primary), var(--shadow-subtle)' : 'var(--shadow-subtle)',
    }),
    badge: {
      fontSize: '10px',
      padding: '2px 8px',
      borderRadius: '4px',
      fontWeight: 600,
      textTransform: 'uppercase' as const,
      letterSpacing: '0.05em',
    },
    btnContinue: (enabled: boolean) => ({
      background: enabled ? 'var(--primary)' : 'var(--border)',
      color: enabled ? '#fff' : 'var(--text-secondary)',
      border: 'none',
      padding: '12px 32px',
      borderRadius: '6px',
      fontSize: '14px',
      fontWeight: 500,
      cursor: enabled ? 'pointer' : 'not-allowed',
      transition: 'all 0.2s',
      display: 'block',
      margin: '0 auto',
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
        <button
          onClick={() => router.push('/dashboard')}
          style={{ fontSize: '13px', color: 'var(--text-secondary)', background: 'none', border: 'none', cursor: 'pointer' }}
        >
          Back to Dashboard
        </button>
      </nav>

      <main style={S.main}>
        <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
          <span style={{ fontSize: '10px', fontWeight: 600, color: 'var(--primary)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Step 1 of 2</span>
          <h1 style={{ fontSize: '2rem', fontWeight: 600, marginTop: '0.5rem', marginBottom: '1rem' }}>Select Exercise</h1>
          <p style={{ color: 'var(--text-secondary)', maxWidth: '500px', margin: '0 auto' }}>
            Choose the specific rehabilitation routine prescribed for your recovery program.
          </p>
        </div>

        {error && (
          <div style={{ background: '#FEF2F2', border: '1px solid #FCA5A5', color: '#991B1B', padding: '12px', borderRadius: '6px', fontSize: '13px', marginBottom: '2rem' }}>
            {error}
          </div>
        )}

        {loading ? (
          <div style={{ textAlign: 'center', padding: '4rem', color: 'var(--text-secondary)' }}>
            Loading recommended exercises...
          </div>
        ) : (
          <div style={S.grid}>
            {exercises.map(ex => (
              <div
                key={ex.id}
                style={S.card(selected?.id === ex.id, false) as any}
                onClick={() => setSelected(ex)}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
                  <div style={{ background: 'var(--bg-medical)', color: 'var(--primary)', padding: '10px', borderRadius: '8px' }}>
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M18 15V18H6V15" />
                      <circle cx="12" cy="7" r="4" />
                    </svg>
                  </div>
                  <span style={{ ...S.badge, background: 'var(--success)', color: '#fff' }}>Available</span>
                </div>
                <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '0.5rem' }}>{ex.name}</h3>
                <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.5, flex: 1 }}>
                  {ex.description}
                </p>
                <div style={{ marginTop: '1.5rem', fontSize: '12px', color: 'var(--primary)', fontWeight: 500, display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="10" />
                    <polyline points="12 6 12 12 16 14" />
                  </svg>
                  {ex.duration_time} min recommended
                </div>
                {selected?.id === ex.id && (
                  <div style={{ position: 'absolute', top: '1rem', right: '1rem', color: 'var(--primary)' }}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                  </div>
                )}
              </div>
            ))}

            {SOON.map(ex => (
              <div key={ex.id} style={S.card(false, true) as any}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
                  <div style={{ background: '#F1F5F9', color: '#94A3B8', padding: '10px', borderRadius: '8px' }}>
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                      <path d="M7 11V7a5 5 0 0 1 10 0v4" />
                    </svg>
                  </div>
                  <span style={{ ...S.badge, background: '#F1F5F9', color: '#64748B' }}>Coming Soon</span>
                </div>
                <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '0.5rem', color: '#94A3B8' }}>{ex.name}</h3>
                <p style={{ fontSize: '13px', color: '#94A3B8', lineHeight: 1.5 }}>
                  {ex.description}
                </p>
              </div>
            ))}
          </div>
        )}

        <div style={{ textAlign: 'center' }}>
          <button style={S.btnContinue(!!selected) as any} disabled={!selected} onClick={handleContinue}>
            Continue
          </button>
          {!selected && (
            <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '1rem' }}>
              Please select an exercise to proceed to session setup.
            </p>
          )}
        </div>
      </main>
    </div>
  );
}