'use client';

import React, { useState, useEffect } from 'react';

interface PoseGuideProps {
  exerciseName: string;
  poseImageSrc: string | string[];
  instructions: string[];
  onStart: () => void;
  onBack?: () => void;
}

export default function PoseGuide({
  exerciseName,
  poseImageSrc,
  instructions,
  onStart,
  onBack,
}: PoseGuideProps) {
  const [countdown, setCountdown] = useState<number | null>(null);
  const [scale, setScale] = useState(1);

  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (countdown !== null && countdown > 0) {
      setScale(1.1);
      const scaleTimeout = setTimeout(() => setScale(1), 200);
      timer = setTimeout(() => {
        setCountdown((prev) => (prev !== null ? prev - 1 : null));
      }, 1000);
      return () => {
        clearTimeout(timer);
        clearTimeout(scaleTimeout);
      };
    } else if (countdown === 0) {
      timer = setTimeout(() => {
        onStart();
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [countdown, onStart]);

  const handleStart = () => {
    setCountdown(5);
  };

  const S = {
    overlay: {
      backgroundColor: 'var(--bg-medical)',
      minHeight: '100vh',
      padding: '2rem',
      display: 'flex',
      flexDirection: 'column' as const,
    },
    nav: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: '2rem',
    },
    container: {
      maxWidth: '1000px',
      margin: '0 auto',
      width: '100%',
      flex: 1,
    },
    grid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))',
      gap: '2rem',
      marginBottom: '3rem',
    },
    card: {
      background: 'var(--surface)',
      border: '1px solid var(--border)',
      borderRadius: '8px',
      padding: '2rem',
      boxShadow: 'var(--shadow-subtle)',
    },
    instructionStep: {
      display: 'flex',
      gap: '1rem',
      alignItems: 'flex-start',
      marginBottom: '1.25rem',
      padding: '1rem',
      background: 'var(--bg-medical)',
      borderRadius: '6px',
      border: '1px solid var(--border)',
    },
    stepNumber: {
      background: 'var(--primary)',
      color: '#fff',
      width: '24px',
      height: '24px',
      borderRadius: '50%',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontSize: '12px',
      fontWeight: 700,
      flexShrink: 0,
    },
    btnPrimary: {
      background: 'var(--primary)',
      color: '#fff',
      border: 'none',
      padding: '16px 48px',
      borderRadius: '6px',
      fontSize: '1.1rem',
      fontWeight: 600,
      cursor: 'pointer',
      boxShadow: 'var(--shadow-subtle)',
      transition: 'all 0.2s',
    }
  };

  if (countdown !== null) {
    return (
      <div style={{ ...S.overlay, justifyContent: 'center', alignItems: 'center' }}>
        <div style={{ 
          fontSize: '8rem', 
          fontWeight: 800, 
          color: 'var(--primary)', 
          transform: `scale(${scale})`, 
          transition: 'transform 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275)' 
        }}>
          {countdown > 0 ? countdown : 'GO'}
        </div>
        <p style={{ fontSize: '1.25rem', color: 'var(--text-secondary)', marginTop: '2rem', fontWeight: 500 }}>
          Position yourself in the frame...
        </p>
      </div>
    );
  }

  const CrossIcon = () => (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 4V20M4 12H20" stroke="currentColor" strokeWidth="3" strokeLinecap="round"/>
    </svg>
  );

  return (
    <div style={S.overlay}>
      <nav style={S.nav}>
        <button 
          onClick={onBack}
          style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '14px', fontWeight: 500 }}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/>
          </svg>
          Back
        </button>
        <div style={{ fontWeight: 600, fontSize: '1.25rem', color: 'var(--primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <CrossIcon />
          <span>PoseCorrect</span>
        </div>
        <div style={{ width: '80px' }} />
      </nav>

      <div style={S.container}>
        <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
          <h1 style={{ fontSize: '2.5rem', fontWeight: 700, marginBottom: '0.5rem' }}>{exerciseName}</h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '1.1rem' }}>Clinical Setup & Instructions</p>
        </div>

        <div style={S.grid}>
          {/* Visual Guide */}
          <div style={S.card}>
            <h2 style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '1.5rem' }}>Visual Reference</h2>
            <div style={{ background: 'var(--bg-medical)', borderRadius: '6px', overflow: 'hidden', aspectRatio: '4/3', display: 'flex' }}>
              {Array.isArray(poseImageSrc) ? (
                poseImageSrc.map((src, i) => (
                  <img key={i} src={src} alt="Guide" style={{ flex: 1, width: '100%', height: '100%', objectFit: 'cover', borderRight: i < poseImageSrc.length - 1 ? '1px solid var(--border)' : 'none' }} />
                ))
              ) : (
                <img src={poseImageSrc} alt="Guide" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
              )}
            </div>
            <p style={{ textAlign: 'center', marginTop: '1.5rem', fontSize: '14px', fontWeight: 500, color: 'var(--primary)' }}>Proper Alignment Mode</p>
          </div>

          {/* Steps */}
          <div style={S.card}>
            <h2 style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '1.5rem' }}>Exercise Protocol</h2>
            <div style={{ display: 'flex', flexDirection: 'column' }}>
              {instructions.map((inst, i) => (
                <div key={i} style={S.instructionStep}>
                  <div style={S.stepNumber}>{i+1}</div>
                  <span style={{ fontSize: '14px', lineHeight: 1.5, color: 'var(--text-primary)' }}>{inst}</span>
                </div>
              ))}
            </div>
            <div style={{ marginTop: 'auto', display: 'flex', gap: '12px', padding: '12px', background: 'rgba(0,119,182,0.05)', borderRadius: '6px', border: '1px solid rgba(0,119,182,0.1)' }}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--primary)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>
              </svg>
              <p style={{ fontSize: '12px', color: 'var(--primary)', margin: 0, fontWeight: 500 }}>Ensure your full body is visible to the AI sensor for accurate tracking.</p>
            </div>
          </div>
        </div>

        <div style={{ textAlign: 'center', paddingBottom: '4rem' }}>
          <button 
            style={S.btnPrimary} 
            onClick={handleStart}
            onMouseOver={(e) => e.currentTarget.style.opacity = '0.9'}
            onMouseOut={(e) => e.currentTarget.style.opacity = '1'}
          >
            Acknowledge & Start
          </button>
        </div>
      </div>
    </div>
  );
}
