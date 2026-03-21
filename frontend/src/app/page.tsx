'use client';

import { useRouter } from 'next/navigation';

export default function HomePage() {
  const router = useRouter();

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
    },
    navLinks: {
      display: 'flex',
      gap: '24px',
      alignItems: 'center',
    },
    navItem: {
      fontSize: '14px',
      color: 'var(--text-secondary)',
      textDecoration: 'none',
      cursor: 'pointer',
      fontWeight: 400,
    },
    btnPrimary: {
      background: 'var(--primary)',
      color: '#fff',
      border: 'none',
      padding: '8px 20px',
      borderRadius: '6px',
      fontSize: '14px',
      fontWeight: 500,
      cursor: 'pointer',
      transition: 'background 0.2s',
    },
    btnSecondary: {
      background: 'transparent',
      border: '1px solid var(--primary)',
      color: 'var(--primary)',
      padding: '8px 20px',
      borderRadius: '6px',
      fontSize: '14px',
      fontWeight: 500,
      cursor: 'pointer',
      transition: 'background 0.2s',
    },
    hero: {
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '5rem 2rem',
      display: 'grid',
      gridTemplateColumns: '1.2fr 1fr',
      gap: '4rem',
      alignItems: 'center',
    },
    h1: {
      fontSize: '3.5rem',
      fontWeight: 700,
      lineHeight: 1.1,
      marginBottom: '1.5rem',
      color: 'var(--text-primary)',
      letterSpacing: '-0.02em',
    },
    subtext: {
      fontSize: '1.1rem',
      color: 'var(--text-secondary)',
      lineHeight: 1.6,
      marginBottom: '2.5rem',
      maxWidth: '500px',
    },
    ctaRow: {
      display: 'flex',
      gap: '12px',
    },
    card: {
      background: 'var(--surface)',
      border: '1px solid var(--border)',
      borderRadius: '8px',
      padding: '1.5rem',
      boxShadow: 'var(--shadow-subtle)',
    },
    sectionTitle: {
      fontSize: '2rem',
      fontWeight: 600,
      textAlign: 'center' as const,
      marginBottom: '3rem',
    },
    featureGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
      gap: '2rem',
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '4rem 2rem',
    },
    stepGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
      gap: '1.5rem',
      maxWidth: '1000px',
      margin: '0 auto',
      padding: '4rem 2rem',
    },
    footer: {
      padding: '2rem',
      textAlign: 'center' as const,
      borderTop: '1px solid var(--border)',
      background: 'var(--surface)',
      fontSize: '12px',
      color: 'var(--text-secondary)',
    }
  };

  const CrossIcon = () => (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 4V20M4 12H20" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );

  return (
    <div style={S.page}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Inter', sans-serif; }
        .btn-hover:hover { opacity: 0.9; }
        .card-accent { border-left: 4px solid var(--primary) !important; }
        .label-upper { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-secondary); margin-bottom: 4px; }
      `}</style>

      {/* Navbar */}
      <nav style={S.nav}>
        <div style={S.logo}>
          <CrossIcon />
          <span>PoseCorrect</span>
        </div>
        <div style={S.navLinks}>
          <span style={S.navItem}>Programs</span>
          <span style={S.navItem}>Clinicians</span>
          <span style={S.navItem}>About</span>
          <button style={S.btnSecondary} onClick={() => router.push('/login')}>Log in</button>
          <button style={S.btnPrimary} onClick={() => router.push('/register')}>Sign up</button>
        </div>
      </nav>

      {/* Hero Section */}
      <section style={S.hero}>
        <div>
          <div className="label-upper" style={{ color: 'var(--primary)', marginBottom: '1rem' }}>Clinical-Grade AI Analysis</div>
          <h1 style={S.h1}>Precision guidance for your recovery.</h1>
          <p style={S.subtext}>
            Our AI-powered rehabilitation platform provides real-time biomechanical feedback to ensure every movement is performed safely and effectively.
          </p>
          <div style={S.ctaRow}>
            <button style={S.btnPrimary} onClick={() => router.push('/register')}>Get Started</button>
            <button style={S.btnSecondary} onClick={() => router.push('/login')}>Patient Portal</button>
          </div>
        </div>
        <div style={{ ...S.card, className: 'card-accent' } as any}>
          <div className="label-upper">Compliance Monitoring</div>
          <div style={{ fontSize: '2.5rem', fontWeight: 700, color: 'var(--primary)', marginBottom: '1.5rem' }}>87% Accuracy</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {['Shoulder Extension', 'Elbow Flexion', 'Spine Alignment'].map((label, i) => (
              <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                    <span>{label}</span>
                    <span style={{ fontWeight: 600 }}>{92 - i * 5}%</span>
                  </div>
                  <div style={{ height: '4px', background: 'var(--bg-medical)', borderRadius: '2px', overflow: 'hidden' }}>
                    <div style={{ width: `${92 - i * 5}%`, height: '100%', background: 'var(--primary)' }} />
                  </div>
                </div>
                <div style={{ width: '8px', height: '8px', background: 'var(--success)', borderRadius: '50%' }} />
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Stats Divider */}
      <div style={{ background: 'var(--surface)', borderTop: '1px solid var(--border)', borderBottom: '1px solid var(--border)', padding: '2.5rem' }}>
        <div style={{ maxWidth: '1000px', margin: '0 auto', display: 'flex', justifyContent: 'space-between', textAlign: 'center' }}>
          {[
            ['33', 'Keypoints'],
            ['98%', 'Accuracy'],
            ['Live', 'Feedback']
          ].map(([val, label]) => (
            <div key={label}>
              <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--primary)' }}>{val}</div>
              <div className="label-upper" style={{ marginTop: '4px' }}>{label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Features */}
      <section style={S.featureGrid}>
        {[
          ['Real-time tracking', 'Biolandmark mapping via device camera for instantaneous joint angle analysis.'],
          ['Clinical reporting', 'Objective data collection for tracking improvements in range of motion and form.'],
          ['Correction cues', 'Guided audio and visual notifications to prevent common injury-prone movements.'],
          ['Secure records', 'Fully encrypted session history accessible only via authenticated practitioner/patient portal.'],
        ].map(([title, desc]) => (
          <div key={title} style={S.card}>
            <div style={{ color: 'var(--primary)', marginBottom: '1rem' }}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
              </svg>
            </div>
            <h3 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '8px' }}>{title}</h3>
            <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.5 }}>{desc}</p>
          </div>
        ))}
      </section>

      {/* Bottom CTA */}
      <section style={{ background: 'var(--surface)', padding: '5rem 2rem', textAlign: 'center' }}>
        <h2 style={{ fontSize: '2.5rem', fontWeight: 700, marginBottom: '1rem' }}>Ready to start your program?</h2>
        <p style={{ ...S.subtext, margin: '0 auto 2.5rem' }}>Join thousands of patients using PoseCorrect to accelerate their recovery with professional clinical guidance.</p>
        <div style={{ ...S.ctaRow, justifyContent: 'center' }}>
          <button style={S.btnPrimary} onClick={() => router.push('/register')}>Create Account</button>
          <button style={S.btnSecondary} onClick={() => router.push('/login')}>Learn More</button>
        </div>
      </section>
    </div>
  );
}