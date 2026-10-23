import React, { useState } from 'react';
import { 
  Award, 
  BarChart2, 
  TrendingUp, 
  Layers, 
  CheckCircle2, 
  BookOpen, 
  ShieldCheck, 
  Target, 
  Zap, 
  Maximize2,
  ChevronDown,
  ChevronUp,
  LineChart
} from 'lucide-react';

export default function ModelEvaluationDashboard() {
  const [activeChart, setActiveChart] = useState('accuracy');
  const [isExpanded, setIsExpanded] = useState(false);

  const charts = [
    { id: 'accuracy', title: 'Before vs After Accuracy', file: '/figures/accuracy_before_after.png' },
    { id: 'f1', title: 'F1-Score Comparison', file: '/figures/f1_comparison.png' },
    { id: 'roc', title: 'ROC Curves & AUC', file: '/figures/roc_curves.png' },
    { id: 'confusion', title: 'Confusion Matrix Shifts', file: '/figures/confusion_matrices_before_after.png' },
    { id: 'loss', title: 'Loss & Convergence Curves', file: '/figures/training_validation_loss_all_models.png' },
    { id: 'gaps', title: 'Research Gap Scores', file: '/figures/research_gap_scores.png' },
    { id: 'master', title: 'Professor Master Dashboard', file: '/figures/professor_dashboard.png' }
  ];

  return (
    <div className="glass-panel" style={{ padding: '1.5rem', marginBottom: '2rem', border: '1px solid var(--border-gold)' }}>
      {/* Header with Best Model Badge & Summary */}
      <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', gap: '1rem', marginBottom: '1.25rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
            <Award size={22} style={{ color: 'var(--gold-primary)' }} />
            <h3 style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--text-main)', margin: 0 }}>
              AI Model Evaluation & Empirical Benchmarks
            </h3>
            <span style={{ 
              background: 'rgba(16, 185, 129, 0.15)', 
              color: '#34d399', 
              fontSize: '0.75rem', 
              fontWeight: 700, 
              padding: '0.2rem 0.6rem', 
              borderRadius: '20px',
              border: '1px solid rgba(16, 185, 129, 0.3)'
            }}>
              PROFESSOR-READY
            </span>
          </div>
          <p style={{ fontSize: '0.88rem', color: 'var(--text-muted)', margin: 0 }}>
            Rigorous cross-corpus empirical evaluation on 5,544 pairs and 734 unseen test samples across 3 distinct neural architectures (MLP with Feature Attention, Deep Feature-Fusion Attention Network, and Siamese Dual-Tower Cross-Attention Network).
          </p>
        </div>

        <button 
          onClick={() => setIsExpanded(!isExpanded)}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            background: 'rgba(197, 155, 76, 0.12)',
            border: '1px solid var(--border-gold)',
            color: 'var(--gold-light)',
            padding: '0.5rem 1rem',
            borderRadius: '10px',
            cursor: 'pointer',
            fontSize: '0.85rem',
            fontWeight: 600,
            transition: 'all 0.2s ease'
          }}
        >
          {isExpanded ? (
            <>Hide Detailed Visual Charts <ChevronUp size={16} /></>
          ) : (
            <>Explore Visual Charts ({charts.length}) <ChevronDown size={16} /></>
          )}
        </button>
      </div>

      {/* Top Level Summary KPI Cards (Requirement 33) */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
        {/* Card 1: Best Model */}
        <div style={{ background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(197, 155, 76, 0.35)', borderRadius: '12px', padding: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--gold-primary)', fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', marginBottom: '0.35rem' }}>
            <Award size={14} /> Best Model
          </div>
          <div style={{ fontSize: '1.05rem', fontWeight: 800, color: '#fff', lineHeight: 1.2, marginBottom: '0.2rem' }}>
            Deep Feature-Fusion Attention Network
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Multi-Modal Group Attention
          </div>
        </div>

        {/* Card 2: Test Accuracy */}
        <div style={{ background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(16, 185, 129, 0.35)', borderRadius: '12px', padding: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: '#34d399', fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', marginBottom: '0.35rem' }}>
            <TrendingUp size={14} /> Test Accuracy
          </div>
          <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#34d399', lineHeight: 1.1, marginBottom: '0.2rem' }}>
            97.28%
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Before: 55.26% (<span style={{ color: '#34d399', fontWeight: 700 }}>+42.0 pp</span>)
          </div>
        </div>

        {/* Card 3: F1-Score */}
        <div style={{ background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(168, 85, 247, 0.35)', borderRadius: '12px', padding: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: '#c084fc', fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', marginBottom: '0.35rem' }}>
            <Zap size={14} /> Test F1-Score
          </div>
          <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#c084fc', lineHeight: 1.1, marginBottom: '0.2rem' }}>
            0.9602
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Precision: 98.8% | Recall: 93.4%
          </div>
        </div>

        {/* Card 4: Research Papers */}
        <div style={{ background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(59, 130, 246, 0.35)', borderRadius: '12px', padding: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: '#60a5fa', fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', marginBottom: '0.35rem' }}>
            <BookOpen size={14} /> Research Papers
          </div>
          <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#60a5fa', lineHeight: 1.1, marginBottom: '0.2rem' }}>
            2,199
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            714 Relevant (OpenAlex API)
          </div>
        </div>

        {/* Card 5: Patent Prior-Art */}
        <div style={{ background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(245, 158, 11, 0.35)', borderRadius: '12px', padding: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: '#fbbf24', fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', marginBottom: '0.35rem' }}>
            <ShieldCheck size={14} /> Patents Ingested
          </div>
          <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#fbbf24', lineHeight: 1.1, marginBottom: '0.2rem' }}>
            343
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            241 Relevant (EPO OPS API)
          </div>
        </div>

        {/* Card 6: Research Gaps */}
        <div style={{ background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(239, 68, 68, 0.35)', borderRadius: '12px', padding: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: '#f87171', fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', marginBottom: '0.35rem' }}>
            <Target size={14} /> Research Gaps
          </div>
          <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#f87171', lineHeight: 1.1, marginBottom: '0.2rem' }}>
            6 Gaps
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            100% Evidence Grounded
          </div>
        </div>
      </div>

      {/* 3-Model Before vs After Benchmark Table */}
      <div style={{ overflowX: 'auto', marginBottom: isExpanded ? '1.5rem' : '0' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem', textAlign: 'left' }}>
          <thead>
            <tr style={{ background: 'rgba(15, 23, 42, 0.85)', borderBottom: '1px solid var(--border-gold)' }}>
              <th style={{ padding: '0.65rem 0.85rem', color: 'var(--gold-light)' }}>Architecture</th>
              <th style={{ padding: '0.65rem 0.85rem', color: 'var(--text-muted)' }}>Before Acc</th>
              <th style={{ padding: '0.65rem 0.85rem', color: 'var(--text-muted)' }}>After Acc</th>
              <th style={{ padding: '0.65rem 0.85rem', color: '#34d399' }}>Accuracy Gain</th>
              <th style={{ padding: '0.65rem 0.85rem', color: 'var(--text-muted)' }}>Before F1</th>
              <th style={{ padding: '0.65rem 0.85rem', color: 'var(--text-muted)' }}>After F1</th>
              <th style={{ padding: '0.65rem 0.85rem', color: 'var(--gold-light)' }}>ROC-AUC</th>
              <th style={{ padding: '0.65rem 0.85rem', color: 'var(--text-muted)' }}>Status</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
              <td style={{ padding: '0.65rem 0.85rem', fontWeight: 600, color: '#fff' }}>Model 1: Multi-Layer Perceptron (MLP) with Feature Attention</td>
              <td style={{ padding: '0.65rem 0.85rem', color: 'var(--text-muted)' }}>56.58%</td>
              <td style={{ padding: '0.65rem 0.85rem', color: '#fff', fontWeight: 700 }}>94.96%</td>
              <td style={{ padding: '0.65rem 0.85rem', color: '#34d399', fontWeight: 700 }}>+38.38 pp ▲</td>
              <td style={{ padding: '0.65rem 0.85rem', color: 'var(--text-muted)' }}>0.5479</td>
              <td style={{ padding: '0.65rem 0.85rem', color: '#fff' }}>0.9287</td>
              <td style={{ padding: '0.65rem 0.85rem', color: 'var(--gold-light)' }}>0.9707</td>
              <td style={{ padding: '0.65rem 0.85rem', color: '#34d399' }}>Early Stop @ Ep 45</td>
            </tr>

            <tr style={{ background: 'rgba(37, 99, 235, 0.1)', borderBottom: '1px solid rgba(37, 99, 235, 0.3)' }}>
              <td style={{ padding: '0.65rem 0.85rem', fontWeight: 700, color: '#60a5fa', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Award size={15} style={{ color: 'var(--gold-primary)' }} /> Model 2: Deep Feature-Fusion Attention Network
              </td>
              <td style={{ padding: '0.65rem 0.85rem', color: 'var(--text-muted)' }}>55.26%</td>
              <td style={{ padding: '0.65rem 0.85rem', color: '#34d399', fontWeight: 800 }}>97.28%</td>
              <td style={{ padding: '0.65rem 0.85rem', color: '#34d399', fontWeight: 800 }}>+42.02 pp ▲</td>
              <td style={{ padding: '0.65rem 0.85rem', color: 'var(--text-muted)' }}>0.5854</td>
              <td style={{ padding: '0.65rem 0.85rem', color: '#c084fc', fontWeight: 800 }}>0.9602</td>
              <td style={{ padding: '0.65rem 0.85rem', color: 'var(--gold-light)', fontWeight: 800 }}>0.9706</td>
              <td style={{ padding: '0.65rem 0.85rem', color: '#34d399', fontWeight: 700 }}>★ BEST GENERALIZER</td>
            </tr>

            <tr>
              <td style={{ padding: '0.65rem 0.85rem', fontWeight: 600, color: '#fff' }}>Model 3: Siamese Dual-Tower Network with Cross-Attention</td>
              <td style={{ padding: '0.65rem 0.85rem', color: 'var(--text-muted)' }}>71.05%</td>
              <td style={{ padding: '0.65rem 0.85rem', color: '#fff', fontWeight: 700 }}>92.37%</td>
              <td style={{ padding: '0.65rem 0.85rem', color: '#34d399', fontWeight: 700 }}>+21.32 pp ▲</td>
              <td style={{ padding: '0.65rem 0.85rem', color: 'var(--text-muted)' }}>0.7600</td>
              <td style={{ padding: '0.65rem 0.85rem', color: '#fff' }}>0.8951</td>
              <td style={{ padding: '0.65rem 0.85rem', color: 'var(--gold-light)', fontWeight: 700 }}>0.9817</td>
              <td style={{ padding: '0.65rem 0.85rem', color: '#34d399' }}>Early Stop @ Ep 14</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Expandable Visual Charts Viewer */}
      {isExpanded && (
        <div style={{ marginTop: '1.5rem', paddingTop: '1.25rem', borderTop: '1px solid rgba(255, 255, 255, 0.08)' }}>
          {/* Chart Tabs */}
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '1.25rem' }}>
            {charts.map((c) => (
              <button
                key={c.id}
                onClick={() => setActiveChart(c.id)}
                style={{
                  background: activeChart === c.id ? 'var(--gold-gradient)' : 'rgba(255, 255, 255, 0.05)',
                  color: activeChart === c.id ? '#07090e' : 'var(--text-muted)',
                  border: `1px solid ${activeChart === c.id ? 'var(--gold-primary)' : 'rgba(255, 255, 255, 0.1)'}`,
                  padding: '0.45rem 0.85rem',
                  borderRadius: '8px',
                  fontSize: '0.8rem',
                  fontWeight: 700,
                  cursor: 'pointer',
                  transition: 'all 0.15s ease'
                }}
              >
                {c.title}
              </button>
            ))}
          </div>

          {/* Active Chart Display */}
          <div style={{ background: '#ffffff', borderRadius: '12px', padding: '1rem', border: '1px solid var(--border-gold)', display: 'flex', justifyContent: 'center' }}>
            <img 
              src={charts.find(c => c.id === activeChart)?.file} 
              alt={charts.find(c => c.id === activeChart)?.title}
              style={{ maxWidth: '100%', maxHeight: '580px', objectFit: 'contain', borderRadius: '8px' }}
              loading="lazy"
            />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.65rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            <span>Publication Resolution: 300 DPI High-Definition Academic Chart</span>
            <a 
              href={charts.find(c => c.id === activeChart)?.file} 
              target="_blank" 
              rel="noopener noreferrer" 
              style={{ color: 'var(--gold-light)', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '0.3rem' }}
            >
              Open Full Resolution <Maximize2 size={13} />
            </a>
          </div>
        </div>
      )}
    </div>
  );
}
