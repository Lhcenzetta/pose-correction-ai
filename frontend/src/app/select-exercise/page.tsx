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

function exerciseIcon(name = '') {
  const n = name.toLowerCase();
  if (n.includes('shoulder')) return '💪';
  if (n.includes('arm')) return '🦾';
  if (n.includes('squat')) return '🏋️';
  if (n.includes('lunge')) return '🦵';
  return '🏃';
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
        setError('Could not load exercises. Is your API running?');
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
      fontFamily: "'DM Sans', sans-serif",
      background: '#f7f9f7',
      minHeight: '100vh',
    } as React.CSSProperties,
    nav: {
      background: '#fff',
      borderBottom: '1px solid #e3ede5',
      padding: '1rem 2rem',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      position: 'sticky' as const,
      top: 0,
      zIndex: 10,
    },
    main: {
      maxWidth: 700,
      margin: '0 auto',
      padding: '3rem 2rem',
      display: 'flex',
      flexDirection: 'column' as const,
      alignItems: 'center',
    },
    grid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(2, 1fr)',
      gap: '1rem',
      width: '100%',
      marginBottom: '2rem',
    },
    card: (isSelected: boolean, disabled: boolean): React.CSSProperties => ({
      background: isSelected ? '#f0f9f3' : '#fff',
      border: `1.5px solid ${isSelected ? '#1a6640' : '#e3ede5'}`,
      borderRadius: 18,
      padding: '1.5rem',
      cursor: disabled ? 'not-allowed' : 'pointer',
      opacity: disabled ? 0.5 : 1,
      transition: 'all 0.2s',
      boxShadow: isSelected ? '0 0 0 3px rgba(26,102,64,0.08)' : 'none',
    }),
    btnContinue: (enabled: boolean): React.CSSProperties => ({
      background: enabled ? '#1a6640' : '#e3ede5',
      color: enabled ? '#fff' : '#b5cfb9',
      border: 'none',
      padding: '14px 48px',
      borderRadius: 100,
      fontSize: 15,
      fontWeight: 500,
      fontFamily: "'DM Sans', sans-serif",
      cursor: enabled ? 'pointer' : 'not-allowed',
      transition: 'all 0.2s',
    }),
  };

  return (
    <div style={S.page}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,300;0,400;0,600;1,300&family=DM+Sans:wght@300;400;500&display=swap');
        .ex-card:hover { border-color: #9dc9aa !important; transform: translateY(-2px); box-shadow: 0 4px 20px rgba(26,102,64,0.08) !important; }
        .ex-card-disabled:hover { transform: none !important; }
      `}</style>

      {/* Navbar */}
      <nav style={S.nav}>
        <div style={{ fontFamily: "'Fraunces', serif", fontWeight: 600, fontSize: 18, color: '#1a6640' }}>
          Pose<span style={{ color: '#b5cfb9' }}>Correct</span>
        </div>
        <button
          onClick={() => router.push('/dashboard')}
          style={{ fontSize: 12, color: '#8aaa90', background: 'none', border: 'none', cursor: 'pointer', fontFamily: "'DM Sans', sans-serif" }}
        >
          ← Back to dashboard
        </button>
      </nav>

      <main style={S.main}>
        {/* Header */}
        <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.1em', color: '#1a6640', marginBottom: 6 }}>
          Step 1 of 2
        </div>
        <h1 style={{ fontFamily: "'Fraunces', serif", fontSize: '2rem', fontWeight: 600, color: '#0f1f13', letterSpacing: -1, textAlign: 'center', marginBottom: 8 }}>
          Select exercise
        </h1>
        <p style={{ fontSize: 13, color: '#8aaa90', fontWeight: 300, textAlign: 'center', marginBottom: '2.5rem' }}>
          Choose the exercise prescribed by your doctor or physiotherapist
        </p>

        {/* Error */}
        {error && (
          <div style={{ background: '#fef3e2', border: '1px solid #f5c4a0', borderRadius: 10, padding: '10px 14px', fontSize: 12, color: '#b06a00', marginBottom: '1.5rem', width: '100%' }}>
            {error}
          </div>
        )}

        {/* Grid */}
        {loading ? (
          <div style={{ textAlign: 'center', padding: '3rem', color: '#b5cfb9', fontSize: 13 }}>
            Loading exercises...
          </div>
        ) : (
          <div style={S.grid}>

            {/* Available — from API */}
            {exercises.map(ex => (
              <div
                key={ex.id}
                className="ex-card"
                style={S.card(selected?.id === ex.id, false)}
                onClick={() => setSelected(ex)}
              >
                <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: '1rem' }}>
                  <div style={{ width: 46, height: 46, background: selected?.id === ex.id ? '#d0ebd8' : '#e8f4ec', borderRadius: 14, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 22 }}>
                    {exerciseIcon(ex.name)}
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ fontSize: 10, padding: '3px 10px', borderRadius: 100, fontWeight: 500, background: '#e8f4ec', color: '#1a6640' }}>
                      Available
                    </span>
                    <div style={{ width: 20, height: 20, borderRadius: '50%', border: `1.5px solid ${selected?.id === ex.id ? '#1a6640' : '#c3d9c7'}`, background: selected?.id === ex.id ? '#1a6640' : 'transparent', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, color: selected?.id === ex.id ? '#fff' : 'transparent', transition: 'all 0.2s', flexShrink: 0 }}>
                      ✓
                    </div>
                  </div>
                </div>
                <div style={{ fontSize: 15, fontWeight: 500, color: '#0f1f13', marginBottom: 6 }}>{ex.name}</div>
                <div style={{ fontSize: 12, color: '#7a9a80', lineHeight: 1.6, fontWeight: 300 }}>
                  {ex.description?.slice(0, 90)}{ex.description?.length > 90 ? '...' : ''}
                </div>
                {ex.duration_time > 0 && (
                  <div style={{ fontSize: 11, color: '#8aaa90', marginTop: 10 }}>⏱ {ex.duration_time} min recommended</div>
                )}
              </div>
            ))}

            {/* Soon — static placeholders */}
            {SOON.map(ex => (
              <div key={ex.id} className="ex-card ex-card-disabled" style={S.card(false, true)}>
                <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: '1rem' }}>
                  <div style={{ width: 46, height: 46, background: '#e8f4ec', borderRadius: 14, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 22 }}>
                    {exerciseIcon(ex.name)}
                  </div>
                  <span style={{ fontSize: 10, padding: '3px 10px', borderRadius: 100, fontWeight: 500, background: '#f3f3f3', color: '#8aaa90' }}>
                    Soon
                  </span>
                </div>
                <div style={{ fontSize: 15, fontWeight: 500, color: '#0f1f13', marginBottom: 6 }}>{ex.name}</div>
                <div style={{ fontSize: 12, color: '#7a9a80', lineHeight: 1.6, fontWeight: 300 }}>{ex.description}</div>
              </div>
            ))}

          </div>
        )}

        {/* CTA */}
        <button style={S.btnContinue(!!selected)} disabled={!selected} onClick={handleContinue}>
          Continue →
        </button>
        <span style={{ fontSize: 12, color: '#b5cfb9', marginTop: 10, fontWeight: 300 }}>
          {selected ? `"${selected.name}" selected` : 'Select an exercise to continue'}
        </span>
      </main>
    </div>
  );
}