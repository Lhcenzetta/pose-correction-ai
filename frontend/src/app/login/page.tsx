'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail]       = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState('');

  const S = {
    page: {
      backgroundColor: 'var(--bg-medical)',
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '2rem',
    },
    card: {
      background: 'var(--surface)',
      border: '1px solid var(--border)',
      borderRadius: '8px',
      padding: '2.5rem',
      width: '100%',
      maxWidth: '400px',
      boxShadow: 'var(--shadow-subtle)',
    },
    label: {
      display: 'block',
      fontSize: '10px',
      fontWeight: 600,
      textTransform: 'uppercase' as const,
      letterSpacing: '0.05em',
      color: 'var(--text-secondary)',
      marginBottom: '6px',
    },
    input: {
      width: '100%',
      background: '#fff',
      border: '1px solid var(--border)',
      borderRadius: '6px',
      padding: '10px 14px',
      fontSize: '14px',
      color: 'var(--text-primary)',
      outline: 'none',
      marginBottom: '1rem',
      transition: 'border-color 0.2s',
    },
    btnPrimary: {
      width: '100%',
      background: 'var(--primary)',
      color: '#fff',
      border: 'none',
      padding: '12px',
      borderRadius: '6px',
      fontSize: '14px',
      fontWeight: 500,
      cursor: 'pointer',
      transition: 'opacity 0.2s',
      marginTop: '1rem',
    },
    logo: {
      textAlign: 'center' as const,
      fontWeight: 700,
      fontSize: '1.5rem',
      color: 'var(--primary)',
      marginBottom: '2rem',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '8px',
      cursor: 'pointer',
    }
  };

  async function handleLogin() {
    if (!email.trim() || !password) {
      setError('Please fill in all fields.');
      return;
    }
    setLoading(true);
    setError('');
    try {
      const res = await fetch(`${API}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: email.trim(), password }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Invalid email or password.');
      localStorage.setItem('access_token', data.access_token);
      router.push('/dashboard');
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={S.page}>
      <style>{`
        input:focus { border-color: var(--primary) !important; }
        .btn-hover:hover { opacity: 0.9; }
      `}</style>
      <div style={S.card}>
        <div style={S.logo} onClick={() => router.push('/')}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 4V20M4 12H20" stroke="currentColor" strokeWidth="3" strokeLinecap="round"/>
          </svg>
          <span>PoseCorrect</span>
        </div>

        <h1 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '0.5rem', textAlign: 'center' }}>Patient Login</h1>
        <p style={{ fontSize: '13px', color: 'var(--text-secondary)', textAlign: 'center', marginBottom: '2rem' }}>
          Access your rehabilitation program
        </p>

        {error && (
          <div style={{ background: '#FEF2F2', border: '1px solid #FCA5A5', color: '#991B1B', padding: '10px', borderRadius: '6px', fontSize: '12px', marginBottom: '1.5rem' }}>
            {error}
          </div>
        )}

        <div>
          <label style={S.label}>Email Address</label>
          <input
            type="email"
            placeholder="name@example.com"
            value={email}
            onChange={e => setEmail(e.target.value)}
            style={S.input}
          />
        </div>

        <div>
          <label style={S.label}>Password</label>
          <input
            type="password"
            placeholder="••••••••"
            value={password}
            onChange={e => setPassword(e.target.value)}
            style={S.input}
          />
        </div>

        <button
          className="btn-hover"
          style={S.btnPrimary}
          disabled={loading}
          onClick={handleLogin}
        >
          {loading ? 'Authenticating...' : 'Sign In'}
        </button>

        <p style={{ textAlign: 'center', fontSize: '13px', marginTop: '1.5rem', color: 'var(--text-secondary)' }}>
          Don't have an account?{' '}
          <Link href="/register" style={{ color: 'var(--primary)', fontWeight: 500, textDecoration: 'none' }}>Register</Link>
        </p>
      </div>
    </div>
  );
}