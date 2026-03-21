'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function SessionRedirect() {
  const router = useRouter();

  useEffect(() => {
    const raw = localStorage.getItem('active_session');
    if (raw) {
      try {
        const parsed = JSON.parse(raw);
        if (parsed.session_id) {
          router.push(`/session/${parsed.session_id}`);
          return;
        }
      } catch (e) {
        console.error('Session data corrupt:', e);
      }
    }
    router.push('/select-exercise');
  }, [router]);

  return (
    <div style={{ 
      display: 'flex', 
      flexDirection: 'column',
      alignItems: 'center', 
      justifyContent: 'center', 
      height: '100vh', 
      backgroundColor: 'var(--bg-medical)',
      color: 'var(--text-secondary)',
      gap: '1.5rem'
    }}>
      <div style={{ 
        width: '32px', 
        height: '32px', 
        border: '3px solid var(--border)', 
        borderTopColor: 'var(--primary)', 
        borderRadius: '50%',
        animation: 'spin 0.8s cubic-bezier(0.4, 0, 0.2, 1) infinite'
      }} />
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
      `}</style>
      <div style={{ textAlign: 'center' }}>
        <span style={{ display: 'block', fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '4px' }}>Securing Clinical Link...</span>
        <span style={{ display: 'block', fontSize: '12px', fontWeight: 400 }}>Redirecting to active protocol</span>
      </div>
    </div>
  );
}
