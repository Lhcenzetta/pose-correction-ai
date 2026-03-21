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
      maxWidth: '460px',
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
    input: (hasError: boolean) => ({
      width: '100%',
      background: '#fff',
      border: `1px solid ${hasError ? '#EF4444' : 'var(--border)'}`,
      borderRadius: '6px',
      padding: '10px 14px',
      fontSize: '14px',
      color: 'var(--text-primary)',
      outline: 'none',
      marginBottom: '1rem',
      transition: 'border-color 0.2s',
    }),
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

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: value }));
    setFieldErrors(prev => ({ ...prev, [name]: '' }));
    setError('');
  }

  function validate(): boolean {
    const errs: Record<string, string> = {};
    if (!form.first_name.trim()) errs.first_name = 'Required';
    if (!form.last_name.trim())  errs.last_name  = 'Required';
    if (!form.email.trim())      errs.email      = 'Required';
    else if (!/\S+@\S+\.\S+/.test(form.email)) errs.email = 'Invalid email';
    if (!form.password)          errs.password   = 'Required';
    else if (form.password.length < 6) errs.password = 'Min. 6 chars';
    if (form.password !== form.confirm_password) errs.confirm_password = 'No match';
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

        <h1 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '0.5rem', textAlign: 'center' }}>Create Patient Account</h1>
        <p style={{ fontSize: '13px', color: 'var(--text-secondary)', textAlign: 'center', marginBottom: '2rem' }}>
          Start your guided recovery program
        </p>

        {error && (
          <div style={{ background: '#FEF2F2', border: '1px solid #FCA5A5', color: '#991B1B', padding: '10px', borderRadius: '6px', fontSize: '12px', marginBottom: '1.5rem' }}>
            {error}
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
          <div>
            <label style={S.label}>First Name</label>
            <input name="first_name" type="text" placeholder="John" value={form.first_name} onChange={handleChange} style={S.input(!!fieldErrors.first_name) as any} />
          </div>
          <div>
            <label style={S.label}>Last Name</label>
            <input name="last_name" type="text" placeholder="Doe" value={form.last_name} onChange={handleChange} style={S.input(!!fieldErrors.last_name) as any} />
          </div>
        </div>

        <div>
          <label style={S.label}>Email Address</label>
          <input name="email" type="email" placeholder="name@example.com" value={form.email} onChange={handleChange} style={S.input(!!fieldErrors.email) as any} />
        </div>

        <div>
          <label style={S.label}>Password</label>
          <input name="password" type="password" placeholder="••••••••" value={form.password} onChange={handleChange} style={S.input(!!fieldErrors.password) as any} />
        </div>

        <div>
          <label style={S.label}>Confirm Password</label>
          <input name="confirm_password" type="password" placeholder="••••••••" value={form.confirm_password} onChange={handleChange} style={S.input(!!fieldErrors.confirm_password) as any} />
        </div>

        <button className="btn-hover" style={S.btnPrimary} disabled={loading} onClick={handleSubmit}>
          {loading ? 'Creating account...' : 'Create Account'}
        </button>

        <p style={{ textAlign: 'center', fontSize: '13px', marginTop: '1.5rem', color: 'var(--text-secondary)' }}>
          Already have an account?{' '}
          <Link href="/login" style={{ color: 'var(--primary)', fontWeight: 500, textDecoration: 'none' }}>Sign In</Link>
        </p>
      </div>
    </div>
  );
}