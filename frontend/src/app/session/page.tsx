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
        console.error('Failed to parse active_session', e);
      }
    }
    router.push('/select-exercise');
  }, [router]);

  return (
    <div style={{ 
      display: 'flex', 
      alignItems: 'center', 
      justifyContent: 'center', 
      height: '100vh', 
      fontFamily: "'DM Sans', sans-serif",
      color: '#8aaa90' 
    }}>
      Redirecting to session...
    </div>
  );
}
