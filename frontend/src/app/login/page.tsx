'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// ── Move static styles outside component so they never recreate ──
const inputStyle: React.CSSProperties = {
  width: '100%',
  background: '#f7f9f7',
  border: '1.5px solid #e3ede5',
  borderRadius: 12,
  padding: '12px 16px',
  fontSize: 14,
  fontFamily: "'DM Sans', sans-serif",
  color: '#0f1f13',
  outline: 'none',
  boxSizing: 'border-box',
  transition: 'border-color 0.2s',
};

const labelStyle: React.CSSProperties = {
  display: 'block',
  fontSize: 12,
  fontWeight: 500,
  color: '#3a5a42',
  marginBottom: 6,
};

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail]       = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState('');

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

      if (!res.ok) {
        throw new Error(data.detail || 'Invalid email or password.');
      }

      localStorage.setItem('access_token', data.access_token);
      router.push('/dashboard');
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ fontFamily: "'DM Sans', sans-serif", background: '#f7f9f7', minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '2rem' }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,300;0,400;0,600;1,300&family=DM+Sans:wght@300;400;500&display=swap');
        input:focus { border-color: #1a6640 !important; box-shadow: 0 0 0 3px rgba(26,102,64,0.08); }
      `}</style>

      <div style={{ background: '#fff', border: '1px solid #e3ede5', borderRadius: 24, padding: '2.5rem', width: '100%', maxWidth: 420, boxShadow: '0 8px 40px rgba(26,102,64,0.07)' }}>

        {/* Back */}
        <button
          onClick={() => router.push('/')}
          style={{ background: 'none', border: 'none', fontSize: 12, color: '#8aaa90', cursor: 'pointer', marginBottom: '1.75rem', fontFamily: "'DM Sans', sans-serif", padding: 0 }}
        >
          ← Back to home
        </button>

        {/* Badge */}
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: 7, background: '#e8f4ec', color: '#1a6640', padding: '4px 12px', borderRadius: 100, fontSize: 11, fontWeight: 500, letterSpacing: '0.05em', textTransform: 'uppercase', marginBottom: '1.25rem' }}>
          <div style={{ width: 5, height: 5, borderRadius: '50%', background: '#1a6640' }} />
          Rehabilitation platform
        </div>

        {/* Title */}
        <h1 style={{ fontFamily: "'Fraunces', serif", fontSize: '1.9rem', fontWeight: 600, color: '#0f1f13', letterSpacing: -1, marginBottom: '0.4rem', lineHeight: 1.1 }}>
          Welcome <em style={{ color: '#1a6640', fontStyle: 'italic', fontWeight: 300 }}>back</em>
        </h1>
        <p style={{ fontSize: 13, color: '#8aaa90', marginBottom: '2rem', fontWeight: 300, lineHeight: 1.6 }}>
          Sign in to continue your rehabilitation program.
        </p>

        {/* Error banner */}
        {error && (
          <div style={{ background: '#fef3e2', border: '1px solid #f5c4a0', borderRadius: 10, padding: '10px 14px', fontSize: 12, color: '#b06a00', marginBottom: '1.25rem' }}>
            {error}
          </div>
        )}

        {/* Email */}
        <div style={{ marginBottom: '1rem' }}>
          <label style={labelStyle}>Email address</label>
          <input
            type="email"
            placeholder="lahcen@example.com"
            value={email}
            onChange={e => { setEmail(e.target.value); setError(''); }}
            onKeyDown={e => e.key === 'Enter' && handleLogin()}
            style={inputStyle}
          />
        </div>

        {/* Password */}
        <div style={{ marginBottom: '1.25rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
            <label style={{ ...labelStyle, marginBottom: 0 }}>Password</label>
            <button
              type="button"
              style={{ background: 'none', border: 'none', fontSize: 12, color: '#1a6640', cursor: 'pointer', fontFamily: "'DM Sans', sans-serif", padding: 0, fontWeight: 500 }}
            >
              Forgot password?
            </button>
          </div>
          <input
            type="password"
            placeholder="••••••••"
            value={password}
            onChange={e => { setPassword(e.target.value); setError(''); }}
            onKeyDown={e => e.key === 'Enter' && handleLogin()}
            style={inputStyle}
          />
        </div>

        {/* Remember me */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: '1.5rem' }}>
          <input
            type="checkbox"
            id="remember"
            style={{ accentColor: '#1a6640', width: 14, height: 14, cursor: 'pointer' }}
          />
          <label htmlFor="remember" style={{ fontSize: 12, color: '#8aaa90', cursor: 'pointer' }}>
            Remember me
          </label>
        </div>

        {/* Submit */}
        <button
          type="button"
          disabled={loading}
          onClick={handleLogin}
          style={{
            width: '100%',
            background: loading ? '#e3ede5' : '#1a6640',
            color: loading ? '#b5cfb9' : '#fff',
            border: 'none',
            padding: 14,
            borderRadius: 100,
            fontSize: 15,
            fontWeight: 500,
            fontFamily: "'DM Sans', sans-serif",
            cursor: loading ? 'not-allowed' : 'pointer',
            transition: 'all 0.2s',
          }}
        >
          {loading ? 'Signing in...' : 'Sign in →'}
        </button>

        {/* Register link */}
        <p style={{ textAlign: 'center', fontSize: 13, color: '#8aaa90', marginTop: '1.25rem' }}>
          No account?{' '}
          <Link href="/register" style={{ color: '#1a6640', fontWeight: 500, textDecoration: 'none' }}>
            Create one
          </Link>
        </p>

      </div>
    </div>
  );
}