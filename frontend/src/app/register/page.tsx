'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function RegisterPage() {
  const router = useRouter();
  const [form, setForm] = useState({
    first_name: '',
    last_name: '',
    email: '',
    password: '',
    confirm_password: '',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: value }));
    setFieldErrors(prev => ({ ...prev, [name]: '' }));
    setError('');
  }

  function validate(): boolean {
    const errs: Record<string, string> = {};
    if (!form.first_name.trim()) errs.first_name = 'First name is required';
    if (!form.last_name.trim())  errs.last_name  = 'Last name is required';
    if (!form.email.trim())      errs.email      = 'Email is required';
    else if (!/\S+@\S+\.\S+/.test(form.email)) errs.email = 'Enter a valid email';
    if (!form.password)          errs.password   = 'Password is required';
    else if (form.password.length < 6) errs.password = 'At least 6 characters';
    if (form.password !== form.confirm_password) errs.confirm_password = 'Passwords do not match';
    setFieldErrors(errs);
    return Object.keys(errs).length === 0;
  }

  async function handleSubmit() {
    if (!validate()) return;
    setLoading(true);
    setError('');

    try {
      const res = await fetch(`${API}/Signup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          first_name: form.first_name.trim(),
          last_name:  form.last_name.trim(),
          email:      form.email.trim(),
          password:   form.password,
        }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Registration failed.');

      // Auto-login
      const loginRes = await fetch(`${API}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: form.email.trim(), password: form.password }),
      });

      if (loginRes.ok) {
        const loginData = await loginRes.json();
        localStorage.setItem('access_token', loginData.access_token);
        router.push('/dashboard');
      } else {
        router.push('/login');
      }
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  // ── Shared input style ────────────────────────────────────────
  function inputStyle(hasError: boolean): React.CSSProperties {
    return {
      width: '100%',
      background: '#f7f9f7',
      border: `1.5px solid ${hasError ? '#f5a0a0' : '#e3ede5'}`,
      borderRadius: 12,
      padding: '12px 16px',
      fontSize: 14,
      fontFamily: "'DM Sans', sans-serif",
      color: '#0f1f13',
      outline: 'none',
      boxSizing: 'border-box',
      transition: 'border-color 0.2s',
    };
  }

  const labelStyle: React.CSSProperties = {
    display: 'block',
    fontSize: 12,
    fontWeight: 500,
    color: '#3a5a42',
    marginBottom: 6,
  };

  const fieldErrStyle: React.CSSProperties = {
    fontSize: 11,
    color: '#c0392b',
    marginTop: 4,
  };

  return (
    <div style={{ fontFamily: "'DM Sans', sans-serif", background: '#f7f9f7', minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '2rem' }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,300;0,400;0,600;1,300&family=DM+Sans:wght@300;400;500&display=swap');
        input:focus { border-color: #1a6640 !important; box-shadow: 0 0 0 3px rgba(26,102,64,0.08); }
      `}</style>

      <div style={{ background: '#fff', border: '1px solid #e3ede5', borderRadius: 24, padding: '2.5rem', width: '100%', maxWidth: 460, boxShadow: '0 8px 40px rgba(26,102,64,0.07)' }}>

        {/* Back */}
        <button onClick={() => router.push('/')}
          style={{ background: 'none', border: 'none', fontSize: 12, color: '#8aaa90', cursor: 'pointer', marginBottom: '1.75rem', fontFamily: "'DM Sans', sans-serif", padding: 0 }}>
          ← Back to home
        </button>

        {/* Badge */}
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: 7, background: '#e8f4ec', color: '#1a6640', padding: '4px 12px', borderRadius: 100, fontSize: 11, fontWeight: 500, letterSpacing: '0.05em', textTransform: 'uppercase', marginBottom: '1.25rem' }}>
          <div style={{ width: 5, height: 5, borderRadius: '50%', background: '#1a6640' }} />
          Rehabilitation platform
        </div>

        {/* Title */}
        <h1 style={{ fontFamily: "'Fraunces', serif", fontSize: '1.9rem', fontWeight: 600, color: '#0f1f13', letterSpacing: -1, marginBottom: '0.4rem', lineHeight: 1.1 }}>
          Create your <em style={{ color: '#1a6640', fontStyle: 'italic', fontWeight: 300 }}>account</em>
        </h1>
        <p style={{ fontSize: 13, color: '#8aaa90', marginBottom: '2rem', fontWeight: 300, lineHeight: 1.6 }}>
          Start your rehabilitation program in seconds.
        </p>

        {/* Global error */}
        {error && (
          <div style={{ background: '#fef3e2', border: '1px solid #f5c4a0', borderRadius: 10, padding: '10px 14px', fontSize: 12, color: '#b06a00', marginBottom: '1.25rem' }}>
            {error}
          </div>
        )}

        {/* First + Last name row */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: '1rem' }}>
          <div>
            <label style={labelStyle}>First name</label>
            <input
              name="first_name"
              type="text"
              placeholder="Lahcen"
              value={form.first_name}
              onChange={handleChange}
              onKeyDown={e => e.key === 'Enter' && handleSubmit()}
              style={inputStyle(!!fieldErrors.first_name)}
            />
            {fieldErrors.first_name && <div style={fieldErrStyle}>{fieldErrors.first_name}</div>}
          </div>
          <div>
            <label style={labelStyle}>Last name</label>
            <input
              name="last_name"
              type="text"
              placeholder="Doe"
              value={form.last_name}
              onChange={handleChange}
              onKeyDown={e => e.key === 'Enter' && handleSubmit()}
              style={inputStyle(!!fieldErrors.last_name)}
            />
            {fieldErrors.last_name && <div style={fieldErrStyle}>{fieldErrors.last_name}</div>}
          </div>
        </div>

        {/* Email */}
        <div style={{ marginBottom: '1rem' }}>
          <label style={labelStyle}>Email address</label>
          <input
            name="email"
            type="email"
            placeholder="lahcen@example.com"
            value={form.email}
            onChange={handleChange}
            onKeyDown={e => e.key === 'Enter' && handleSubmit()}
            style={inputStyle(!!fieldErrors.email)}
          />
          {fieldErrors.email && <div style={fieldErrStyle}>{fieldErrors.email}</div>}
        </div>

        {/* Password */}
        <div style={{ marginBottom: '1rem' }}>
          <label style={labelStyle}>Password</label>
          <input
            name="password"
            type="password"
            placeholder="Min. 6 characters"
            value={form.password}
            onChange={handleChange}
            onKeyDown={e => e.key === 'Enter' && handleSubmit()}
            style={inputStyle(!!fieldErrors.password)}
          />
          {fieldErrors.password && <div style={fieldErrStyle}>{fieldErrors.password}</div>}
        </div>

        {/* Confirm password */}
        <div style={{ marginBottom: '1.75rem' }}>
          <label style={labelStyle}>Confirm password</label>
          <input
            name="confirm_password"
            type="password"
            placeholder="••••••••"
            value={form.confirm_password}
            onChange={handleChange}
            onKeyDown={e => e.key === 'Enter' && handleSubmit()}
            style={inputStyle(!!fieldErrors.confirm_password)}
          />
          {fieldErrors.confirm_password && <div style={fieldErrStyle}>{fieldErrors.confirm_password}</div>}
        </div>

        {/* Submit */}
        <button
          disabled={loading}
          onClick={handleSubmit}
          style={{
            width: '100%',
            background: loading ? '#e3ede5' : '#1a6640',
            color: loading ? '#b5cfb9' : '#fff',
            border: 'none', padding: 14, borderRadius: 100,
            fontSize: 15, fontWeight: 500,
            fontFamily: "'DM Sans', sans-serif",
            cursor: loading ? 'not-allowed' : 'pointer',
            transition: 'all 0.2s',
          }}
        >
          {loading ? 'Creating account...' : 'Create account →'}
        </button>

        {/* Login link */}
        <p style={{ textAlign: 'center', fontSize: 13, color: '#8aaa90', marginTop: '1.25rem' }}>
          Already have an account?{' '}
          <Link href="/login" style={{ color: '#1a6640', fontWeight: 500, textDecoration: 'none' }}>Sign in</Link>
        </p>

      </div>
    </div>
  );
}