import React, { useState } from 'react';
import { Sparkles, Send, RefreshCw, Layers } from 'lucide-react';

const PRESETS = [
  {
    label: "AI for Healthcare (Disease Prediction)",
    text: "Artificial Intelligence for Healthcare: Early Disease Prediction and Diagnosis"
  },
  {
    label: "Drug-Target Affinity (GNN + Attention)",
    text: "Using spatial graph neural networks and dynamic transformer attention to predict drug-target binding affinity with explainable molecular substructure alignment under 3D dynamic thermal fluctuations."
  },
  {
    label: "Cryo-EM Ligand Pose Fitting",
    text: "Real-time transformer-assisted small molecule pose optimization in cryo-electron microscopy density maps addressing flexible side-chain conformation limitations in static patent specifications."
  },
  {
    label: "Solid-State Battery Electrolytes",
    text: "High-entropy solid electrolyte ceramic matrices with accelerated lithium-ion conductivity bypassing dendrite formation and patent claim boundary restrictions."
  },
  {
    label: "Quantum High-Throughput Screening",
    text: "Quantum-inspired variational feature maps for virtual screening of large chemical space bypassing NISQ gate noise constraints and patent prior-art barriers."
  }
];

export default function IdeaForm({ onSubmit, loading }) {
  const [ideaText, setIdeaText] = useState(PRESETS[0].text);
  const [topK, setTopK] = useState(10);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!ideaText.trim()) return;
    onSubmit({ idea_text: ideaText, top_k: topK });
  };

  return (
    <div className="glass-panel" style={{ padding: '2.25rem', marginBottom: '2.5rem', borderColor: 'var(--border-gold)' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem', marginBottom: '1.25rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div style={{ width: '38px', height: '38px', borderRadius: '10px', background: 'rgba(197, 155, 76, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--gold-primary)' }}>
            <Sparkles size={20} />
          </div>
          <div>
            <h2 style={{ fontSize: '1.35rem', fontWeight: 800, color: 'var(--text-main)', margin: 0 }}>
              Submit Research Idea or Invention Concept
            </h2>
            <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-muted)' }}>
              Jointly verified against OpenAlex literature, EPO OPS patent claims, and 3D knowledge topology.
            </p>
          </div>
        </div>

        <span className="pill-badge" style={{ padding: '0.35rem 0.85rem', fontSize: '0.75rem' }}>
          <span className="pill-dot"></span> LIVE ENGINE ACTIVE
        </span>
      </div>

      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: '1.25rem' }}>
          <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-main)', marginBottom: '0.5rem', fontWeight: 700, letterSpacing: '0.02em' }}>
            RESEARCH HYPOTHESIS OR PATENT CLAIM HYPOTHESIS:
          </label>
          <textarea
            className="form-input"
            rows={4}
            value={ideaText}
            onChange={(e) => setIdeaText(e.target.value)}
            placeholder="Describe your research idea, proposed methodology, target application area, or technical constraints..."
            required
            style={{ fontSize: '0.95rem', lineHeight: 1.6 }}
          />
        </div>

        {/* Quick Presets */}
        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.6rem', marginBottom: '1.5rem' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-dim)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Quick Presets:
          </span>
          {PRESETS.map((p, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => setIdeaText(p.text)}
              style={{
                background: 'rgba(255, 255, 255, 0.05)',
                border: '1px solid var(--border-subtle)',
                borderRadius: '9999px',
                padding: '0.4rem 0.85rem',
                color: 'var(--text-muted)',
                fontSize: '0.78rem',
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = 'var(--gold-primary)';
                e.currentTarget.style.color = 'var(--gold-light)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = 'var(--border-subtle)';
                e.currentTarget.style.color = 'var(--text-muted)';
              }}
            >
              {p.label}
            </button>
          ))}
        </div>

        {/* Bottom Bar: Retrieval Depth and Action Button */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem', paddingTop: '1.25rem', borderTop: '1px solid var(--border-subtle)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <label style={{ fontSize: '0.85rem', color: 'var(--text-main)', fontWeight: 700 }}>
              Retrieval Depth (Top-K):
            </label>
            <select
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="form-input"
              style={{ width: 'auto', padding: '0.45rem 1rem', fontSize: '0.85rem' }}
            >
              <option value={10}>10 Documents (Standard Precision)</option>
              <option value={20}>20 Documents (Deep Dual-Corpus)</option>
              <option value={30}>30 Documents (Exhaustive Global Scan)</option>
            </select>
          </div>

          <button 
            type="submit" 
            className="btn-gold" 
            disabled={loading}
            style={{ padding: '0.85rem 1.85rem', opacity: loading ? 0.7 : 1 }}
          >
            {loading ? (
              <>
                <RefreshCw className="animate-spin" size={18} />
                Executing Pipeline...
              </>
            ) : (
              <>
                <Send size={18} />
                Run Cross-Corpus Analysis
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
