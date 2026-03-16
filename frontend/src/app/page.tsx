'use client';

import { useRouter } from 'next/navigation';

export default function HomePage() {
  const router = useRouter();

  return (
    <div className="om">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,300;0,400;0,600;1,300&family=DM+Sans:wght@300;400;500&display=swap');
        .om * { box-sizing: border-box; margin: 0; padding: 0; }
        .om {
          font-family: 'DM Sans', sans-serif;
          background: #f7f9f7;
          color: #1a2e1f;
          min-height: 100vh;
        }
        .om-nav {
          display: flex; align-items: center; justify-content: space-between;
          padding: 1.25rem 2.5rem;
          background: #fff;
          border-bottom: 1px solid #e3ede5;
          position: sticky; top: 0; z-index: 10;
        }
        .om-logo {
          font-family: 'Fraunces', serif;
          font-weight: 600; font-size: 20px;
          color: #1a6640; letter-spacing: -0.5px;
        }
        .om-logo span { color: #b5cfb9; }
        .nav-links { display: flex; gap: 10px; align-items: center; }
        .btn-outline {
          background: transparent;
          border: 1px solid #c3d9c7; color: #1a6640;
          padding: 8px 22px; border-radius: 100px;
          font-size: 13px; font-family: 'DM Sans', sans-serif;
          cursor: pointer; transition: all 0.2s;
        }
        .btn-outline:hover { background: #f0f7f1; border-color: #1a6640; }
        .btn-filled {
          background: #1a6640; border: none; color: #fff;
          padding: 9px 22px; border-radius: 100px;
          font-size: 13px; font-family: 'DM Sans', sans-serif;
          cursor: pointer; transition: all 0.2s;
        }
        .btn-filled:hover { background: #155435; transform: translateY(-1px); }
        .om-hero {
          display: grid; grid-template-columns: 1fr 1fr;
          gap: 3rem; align-items: center;
          max-width: 1000px; margin: 0 auto;
          padding: 5rem 2.5rem 4rem;
        }
        @media (max-width: 680px) {
          .om-hero { grid-template-columns: 1fr; }
          .om-hero-visual { display: none; }
        }
        .om-tag {
          display: inline-flex; align-items: center; gap: 7px;
          background: #e8f4ec; color: #1a6640;
          padding: 5px 14px; border-radius: 100px;
          font-size: 11px; font-weight: 500;
          letter-spacing: 0.05em; text-transform: uppercase;
          margin-bottom: 1.5rem;
        }
        .om-tag-line { width: 16px; height: 2px; background: #1a6640; border-radius: 2px; }
        .om-h1 {
          font-family: 'Fraunces', serif; font-weight: 600;
          font-size: clamp(2rem, 4vw, 3rem);
          line-height: 1.15; letter-spacing: -1px;
          color: #0f1f13; margin-bottom: 1.25rem;
        }
        .om-h1 em { color: #1a6640; font-style: italic; font-weight: 300; }
        .om-sub {
          font-size: 15px; color: #5a7a62;
          line-height: 1.75; font-weight: 300;
          margin-bottom: 2rem; max-width: 420px;
        }
        .om-cta-row { display: flex; gap: 10px; flex-wrap: wrap; }
        .btn-hero {
          background: #1a6640; border: none; color: #fff;
          padding: 13px 28px; border-radius: 100px;
          font-size: 14px; font-weight: 500;
          font-family: 'DM Sans', sans-serif;
          cursor: pointer; transition: all 0.2s;
        }
        .btn-hero:hover { background: #155435; transform: translateY(-1px); box-shadow: 0 6px 20px rgba(26,102,64,0.2); }
        .btn-hero-ghost {
          background: transparent; border: 1px solid #c3d9c7; color: #1a6640;
          padding: 13px 28px; border-radius: 100px;
          font-size: 14px; font-family: 'DM Sans', sans-serif;
          cursor: pointer; transition: all 0.2s;
        }
        .btn-hero-ghost:hover { background: #f0f7f1; }
        .visual-card {
          background: #fff; border: 1px solid #e3ede5;
          border-radius: 16px; padding: 1.25rem 1.5rem;
          margin-bottom: 12px;
        }
        .vc-label { font-size: 11px; color: #8aaa90; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }
        .vc-score { font-family: 'Fraunces', serif; font-size: 2.8rem; font-weight: 600; color: #1a6640; letter-spacing: -2px; line-height: 1; }
        .vc-score span { font-size: 1rem; color: #8aaa90; font-family: 'DM Sans', sans-serif; font-weight: 300; }
        .vc-bar-row { display: flex; flex-direction: column; gap: 8px; margin-top: 12px; }
        .vc-bar-item { display: flex; align-items: center; gap: 10px; font-size: 12px; color: #5a7a62; }
        .vc-bar-track { flex: 1; height: 5px; background: #e8f4ec; border-radius: 100px; overflow: hidden; }
        .vc-bar-fill { height: 100%; background: #1a6640; border-radius: 100px; }
        .vc-small { display: flex; gap: 10px; flex-wrap: wrap; }
        .vc-pill { background: #e8f4ec; color: #1a6640; border-radius: 100px; padding: 6px 14px; font-size: 12px; font-weight: 500; }
        .vc-pill.warn { background: #fef3e2; color: #b06a00; }
        .om-divider {
          display: flex; align-items: center; justify-content: center;
          gap: 3rem; flex-wrap: wrap;
          padding: 2rem 2.5rem;
          background: #fff;
          border-top: 1px solid #e3ede5; border-bottom: 1px solid #e3ede5;
        }
        .divider-stat { text-align: center; }
        .ds-num { font-family: 'Fraunces', serif; font-size: 1.75rem; font-weight: 600; color: #1a6640; letter-spacing: -1px; }
        .ds-label { font-size: 12px; color: #8aaa90; margin-top: 2px; }
        .om-features { max-width: 900px; margin: 0 auto; padding: 4rem 2.5rem; }
        .sec-eyebrow { font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; color: #1a6640; margin-bottom: 0.5rem; }
        .sec-title { font-family: 'Fraunces', serif; font-size: clamp(1.4rem, 3vw, 2rem); font-weight: 600; letter-spacing: -0.5px; color: #0f1f13; margin-bottom: 2.5rem; }
        .feat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1.25rem; }
        .feat-card { background: #fff; border: 1px solid #e3ede5; border-radius: 16px; padding: 1.5rem; transition: border-color 0.2s, transform 0.2s; }
        .feat-card:hover { border-color: #9dc9aa; transform: translateY(-2px); }
        .feat-icon-wrap { width: 38px; height: 38px; background: #e8f4ec; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 16px; margin-bottom: 1rem; }
        .feat-title { font-size: 14px; font-weight: 500; color: #0f1f13; margin-bottom: 6px; }
        .feat-desc { font-size: 12px; color: #7a9a80; line-height: 1.65; font-weight: 300; }
        .om-how { background: #fff; border-top: 1px solid #e3ede5; border-bottom: 1px solid #e3ede5; padding: 4rem 2.5rem; }
        .om-how-inner { max-width: 900px; margin: 0 auto; }
        .steps-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem; }
        .step { display: flex; flex-direction: column; gap: 8px; padding: 1.5rem; border-left: 2px solid #e3ede5; transition: border-color 0.2s; }
        .step:hover { border-color: #1a6640; }
        .step-n { font-family: 'Fraunces', serif; font-size: 1.5rem; font-weight: 600; color: #c3d9c7; letter-spacing: -1px; }
        .step-title { font-size: 14px; font-weight: 500; color: #0f1f13; }
        .step-desc { font-size: 12px; color: #7a9a80; line-height: 1.65; font-weight: 300; }
        .om-bottom-cta { max-width: 620px; margin: 0 auto; text-align: center; padding: 5rem 2.5rem; }
        .om-bottom-cta h2 { font-family: 'Fraunces', serif; font-size: clamp(1.6rem, 3vw, 2.4rem); font-weight: 600; letter-spacing: -1px; color: #0f1f13; margin-bottom: 0.75rem; line-height: 1.2; }
        .om-bottom-cta h2 em { color: #1a6640; font-style: italic; font-weight: 300; }
        .om-bottom-cta p { font-size: 14px; color: #7a9a80; margin-bottom: 2rem; line-height: 1.7; font-weight: 300; }
        .bottom-btns { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; }
        .om-footer { text-align: center; padding: 1.5rem; font-size: 11px; color: #c3d9c7; border-top: 1px solid #e3ede5; background: #fff; }
      `}</style>

      {/* Navbar */}
      <nav className="om-nav">
        <div className="om-logo">Pose<span>Correct</span></div>
        <div className="nav-links">
          <button className="btn-outline" onClick={() => router.push('/login')}>Log in</button>
          <button className="btn-filled" onClick={() => router.push('/register')}>Get started</button>
        </div>
      </nav>

      {/* Hero */}
      <section className="om-hero">
        <div>
          <div className="om-tag"><div className="om-tag-line" /> AI rehabilitation platform</div>
          <h1 className="om-h1">Your body deserves <em>precise</em> correction</h1>
          <p className="om-sub">Real-time pose analysis guides every rep of your rehabilitation program — like a clinical expert watching over your shoulder.</p>
          <div className="om-cta-row">
            <button className="btn-hero" onClick={() => router.push('/register')}>Start for free</button>
            <button className="btn-hero-ghost" onClick={() => router.push('/login')}>Sign in</button>
          </div>
        </div>
        <div className="om-hero-visual">
          <div className="visual-card">
            <div className="vc-label">Session accuracy</div>
            <div className="vc-score">87 <span>/ 100</span></div>
            <div className="vc-bar-row">
              <div className="vc-bar-item"><span style={{width:72}}>Spine</span><div className="vc-bar-track"><div className="vc-bar-fill" style={{width:'92%'}} /></div><span>92%</span></div>
              <div className="vc-bar-item"><span style={{width:72}}>Knees</span><div className="vc-bar-track"><div className="vc-bar-fill" style={{width:'78%'}} /></div><span>78%</span></div>
              <div className="vc-bar-item"><span style={{width:72}}>Shoulders</span><div className="vc-bar-track"><div className="vc-bar-fill" style={{width:'88%'}} /></div><span>88%</span></div>
            </div>
          </div>
          <div className="visual-card">
            <div className="vc-label">Feedback</div>
            <div className="vc-small">
              <span className="vc-pill">✓ Good spine alignment</span>
              <span className="vc-pill warn">↑ Raise left knee</span>
            </div>
          </div>
        </div>
      </section>

      {/* Stats */}
      <div className="om-divider">
        {[['33','Keypoints tracked'],['98%','Detection accuracy'],['30+','Exercises'],['Live','Real-time feedback']].map(([num, label]) => (
          <div className="divider-stat" key={label}>
            <div className="ds-num">{num}</div>
            <div className="ds-label">{label}</div>
          </div>
        ))}
      </div>

      {/* Features */}
      <section className="om-features">
        <div className="sec-eyebrow">Features</div>
        <div className="sec-title">Built for serious rehabilitation</div>
        <div className="feat-grid">
          {[
            ['🦴','Skeletal pose detection','Maps 33 body landmarks via camera to analyze joint angles and posture in real time.'],
            ['📈','Accuracy scoring','Each session produces a precise score so progress is measurable and objective.'],
            ['💬','Instant corrections','Audio and visual cues tell you what to fix mid-movement, not after the fact.'],
            ['🗃️','Session history','Every workout is logged with duration, score, and detailed feedback for review.'],
            ['🏥','Rehab exercise library','A curated library of physiotherapy-aligned movements with reference poses.'],
            ['🔐','Private & secure','Your data is protected with JWT authentication. Only you access your sessions.'],
          ].map(([icon, title, desc]) => (
            <div className="feat-card" key={title as string}>
              <div className="feat-icon-wrap">{icon}</div>
              <div className="feat-title">{title}</div>
              <div className="feat-desc">{desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section className="om-how">
        <div className="om-how-inner">
          <div className="sec-eyebrow">How it works</div>
          <div className="sec-title">Simple by design</div>
          <div className="steps-grid">
            {[
              ['01','Create an account','Sign up in under a minute. No equipment beyond your device camera.'],
              ['02','Choose an exercise','Select from the rehab library. The AI loads your reference pose automatically.'],
              ['03','Train with guidance','Get live corrections as you move, then review your full session report.'],
              ['04','Track your progress','Watch your scores improve over time with your personal session history.'],
            ].map(([n, title, desc]) => (
              <div className="step" key={n as string}>
                <div className="step-n">{n}</div>
                <div className="step-title">{title}</div>
                <div className="step-desc">{desc}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Bottom CTA */}
      <section className="om-bottom-cta">
        <h2>Recover smarter, <em>not harder</em></h2>
        <p>Join and let AI guide your rehabilitation with the precision of a clinical professional.</p>
        <div className="bottom-btns">
          <button className="btn-hero" onClick={() => router.push('/register')}>Create free account</button>
          <button className="btn-hero-ghost" onClick={() => router.push('/login')}>I already have an account</button>
        </div>
      </section>

      <footer className="om-footer">PoseCorrect — AI Exercise Correction Platform</footer>
    </div>
  );
}