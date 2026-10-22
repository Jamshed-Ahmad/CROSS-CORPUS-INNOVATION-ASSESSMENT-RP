import React from 'react';
import { ArrowUpRight, ArrowDown, Sparkles, Database, ShieldCheck, Box } from 'lucide-react';

export default function HeroSection({ onRunAssessment, onExplore3D }) {
  return (
    <section className="hero-wrapper" id="about">
      <div className="hero-backdrop-graphic"></div>

      <div style={{ maxWidth: '1100px', margin: '0 auto', textAlign: 'center', padding: '4.5rem 1.5rem 3rem 1.5rem', position: 'relative', zIndex: 2 }}>
        {/* Exact Project Name Pill Badge */}
        <div className="pill-badge">
          <span className="pill-dot"></span>
          AN EVIDENCE-GROUNDED AI FRAMEWORK
        </div>

        {/* Hero Headline matching exact project name */}
        <h1 className="hero-title" style={{ maxWidth: '1050px', margin: '0 auto 1.5rem auto' }}>
          <span className="hero-title-first">Cross-Corpus Innovation Assessment</span>
          <span className="hero-title-gradient">and Recommendation</span>
          <span style={{ display: 'block', fontSize: 'clamp(1.3rem, 2.6vw, 2rem)', fontWeight: 700, color: 'var(--gold-light)', marginTop: '0.6rem', letterSpacing: '-0.01em' }}>
            Using Research Papers & Patents
          </span>
        </h1>

        {/* Hero Subtitle */}
        <p className="hero-description" style={{ marginBottom: '2.5rem' }}>
          An Evidence-Grounded AI Framework bridging peer-reviewed scientific literature (OpenAlex) with patent prior-art claims (European Patent Office - EPO OPS) to uncover unexploited research gaps, legal claim boundaries, and verifiable innovation opportunities.
        </p>

        {/* Dual Call-to-Action Buttons */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '1.25rem', flexWrap: 'wrap', marginBottom: '3.5rem' }}>
          <button 
            onClick={onRunAssessment}
            className="btn-gold"
          >
            RUN LIVE ASSESSMENT <ArrowUpRight size={18} />
          </button>

          <button 
            onClick={onExplore3D}
            className="btn-pill-white"
          >
            VIEW 3D UNIVERSE <ArrowDown size={18} />
          </button>
        </div>

        {/* Quick Highlights Strip */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1.25rem', textAlign: 'left' }}>
          <div className="glass-panel" style={{ padding: '1.25rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
              <div style={{ width: '32px', height: '32px', borderRadius: '8px', background: 'rgba(197, 155, 76, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--gold-primary)' }}>
                <Database size={18} />
              </div>
              <strong style={{ fontSize: '0.95rem' }}>OpenAlex Ingestion</strong>
            </div>
            <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-muted)' }}>
              250M+ scholarly works, abstracts, and citation graph linkages.
            </p>
          </div>

          <div className="glass-panel" style={{ padding: '1.25rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
              <div style={{ width: '32px', height: '32px', borderRadius: '8px', background: 'rgba(168, 85, 247, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent-purple)' }}>
                <ShieldCheck size={18} />
              </div>
              <strong style={{ fontSize: '0.95rem' }}>EPO OPS Patents</strong>
            </div>
            <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-muted)' }}>
              140M+ worldwide patent claims and legal prior-art boundaries.
            </p>
          </div>

          <div className="glass-panel" style={{ padding: '1.25rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
              <div style={{ width: '32px', height: '32px', borderRadius: '8px', background: 'rgba(6, 182, 212, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent-cyan)' }}>
                <Box size={18} />
              </div>
              <strong style={{ fontSize: '0.95rem' }}>3D Animated Universe</strong>
            </div>
            <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-muted)' }}>
              Interactive 360° spatial orbit connecting papers, patents, and gaps.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
