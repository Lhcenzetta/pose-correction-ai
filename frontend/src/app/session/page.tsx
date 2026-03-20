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
      gap: '1rem'
    }}>
      <div style={{ 
        width: '24px', 
        height: '24px', 
        border: '2px solid var(--border)', 
        borderTopColor: 'var(--primary)', 
        borderRadius: '50%',
        animation: 'spin 0.6s linear infinite'
      }} />
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      <span style={{ fontSize: '14px', fontWeight: 500 }}>Redirecting to active protocol...</span>
    </div>
  );
}
