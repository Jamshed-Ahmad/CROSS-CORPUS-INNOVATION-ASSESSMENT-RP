import React, { useState } from 'react';
import {
  BookOpen, ExternalLink, Calendar, User, Database, Download,
  Lock, CheckCircle2, AlertCircle, Quote, Sparkles, Layers
} from 'lucide-react';
import { downloadPaperPdf } from '../services/api';

export default function PaperResults({ papers }) {
  const [downloadingId, setDownloadingId] = useState(null);
  const [downloadFeedback, setDownloadFeedback] = useState({});

  if (!papers || papers.length === 0) {
    return (
      <div className="glass-panel" style={{ padding: '1.5rem', color: 'var(--text-muted)' }}>
        No research papers retrieved.
      </div>
    );
  }

  const handleDownload = async (paper) => {
    if (!paper.pdf_url) return;
    setDownloadingId(paper.paper_id);
    setDownloadFeedback((prev) => ({ ...prev, [paper.paper_id]: null }));

    try {
      await downloadPaperPdf(paper.pdf_url, paper.title);
      setDownloadFeedback((prev) => ({
        ...prev,
        [paper.paper_id]: { type: 'success', message: 'PDF downloaded successfully.' }
      }));
    } catch (err) {
      setDownloadFeedback((prev) => ({
        ...prev,
        [paper.paper_id]: {
          type: 'error',
          message: err.message || 'Direct PDF restricted by publisher. Please use [View Source].'
        }
      }));
    } finally {
      setDownloadingId(null);
    }
  };

  return (
    <div style={{ display: 'grid', gap: '1.5rem' }}>
      {papers.map((paper, idx) => {
        const isOA = Boolean(paper.is_open_access || (paper.open_access_status && paper.open_access_status !== 'closed'));
        const hasPdf = Boolean(paper.pdf_url);
        const sourceUrl = paper.source_url || (paper.doi ? `https://doi.org/${paper.doi}` : null);
        const feedback = downloadFeedback[paper.paper_id];

        return (
          <div
            key={paper.paper_id || idx}
            className="glass-panel paper-card"
            style={{
              padding: '1.75rem',
              position: 'relative',
              borderRadius: '16px',
              border: '1px solid rgba(255, 255, 255, 0.12)'
            }}
          >
            {/* Top row: Badges, Relevance, and Primary Actions */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem', marginBottom: '0.85rem' }}>
              <div>
                <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap', marginBottom: '0.5rem' }}>
                  <span className="tag tag-teal" style={{ fontWeight: 700 }}>
                    #{idx + 1} {paper.paper_id}
                  </span>

                  <span
                    className="tag"
                    style={{
                      background: paper.source_provider === 'openalex' ? 'rgba(59, 130, 246, 0.18)' : 'rgba(255, 255, 255, 0.08)',
                      color: paper.source_provider === 'openalex' ? '#60a5fa' : '#94a3b8',
                      border: '1px solid rgba(255,255,255,0.12)'
                    }}
                  >
                    <Database size={12} style={{ display: 'inline', marginRight: '4px' }} />
                    {paper.source_provider === 'openalex' ? 'OpenAlex API' : 'Academic Fixture'}
                  </span>

                  {isOA ? (
                    <span
                      className="tag"
                      style={{
                        background: 'rgba(16, 185, 129, 0.18)',
                        color: '#34d399',
                        border: '1px solid rgba(16, 185, 129, 0.35)',
                        fontWeight: 650
                      }}
                    >
                      <CheckCircle2 size={12} style={{ display: 'inline', marginRight: '4px' }} />
                      Open Access {paper.open_access_status ? `(${paper.open_access_status})` : ''}
                    </span>
                  ) : (
                    <span
                      className="tag"
                      style={{
                        background: 'rgba(244, 63, 94, 0.15)',
                        color: '#fb7185',
                        border: '1px solid rgba(244, 63, 94, 0.3)'
                      }}
                    >
                      <Lock size={12} style={{ display: 'inline', marginRight: '4px' }} />
                      Paywalled / Subscription
                    </span>
                  )}

                  {paper.relevance_score !== null && paper.relevance_score !== undefined && (
                    <span
                      className="tag"
                      style={{
                        background: 'rgba(217, 119, 6, 0.2)',
                        color: '#f59e0b',
                        border: '1px solid rgba(217, 119, 6, 0.4)',
                        fontWeight: 700
                      }}
                    >
                      <Sparkles size={12} style={{ display: 'inline', marginRight: '4px' }} />
                      Relevance: {Number(paper.relevance_score).toFixed(1)}%
                    </span>
                  )}
                </div>

                <h3 style={{ fontSize: '1.2rem', fontWeight: 750, color: '#fff', lineHeight: 1.35, margin: 0 }}>
                  {paper.title}
                </h3>

                {paper.journal && (
                  <div style={{ fontSize: '0.85rem', color: 'var(--primary-teal)', marginTop: '0.25rem', fontWeight: 600 }}>
                    {paper.journal}
                  </div>
                )}
              </div>

              {/* Action Buttons: DOWNLOAD PAPER + VIEW SOURCE */}
              <div style={{ display: 'flex', gap: '0.65rem', alignItems: 'center', flexWrap: 'wrap' }}>
                {hasPdf ? (
                  <button
                    className="btn-paper-download"
                    onClick={() => handleDownload(paper)}
                    disabled={downloadingId === paper.paper_id}
                    title="Download verified open-access paper PDF"
                  >
                    <Download size={14} />
                    {downloadingId === paper.paper_id ? 'DOWNLOADING...' : 'DOWNLOAD PAPER'}
                  </button>
                ) : (
                  <span
                    className="btn-paper-disabled"
                    title="Direct PDF is restricted by publisher authentication or subscription"
                  >
                    <Lock size={13} style={{ marginRight: '4px' }} />
                    PDF NOT AVAILABLE
                  </span>
                )}

                {sourceUrl && (
                  <a
                    href={sourceUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="btn-paper-source"
                    title="Open publication landing page or DOI resolver"
                  >
                    <ExternalLink size={14} />
                    VIEW SOURCE
                  </a>
                )}
              </div>
            </div>

            {/* Download Status Notification */}
            {feedback && (
              <div
                style={{
                  fontSize: '0.82rem',
                  padding: '0.5rem 0.8rem',
                  borderRadius: '8px',
                  marginBottom: '0.85rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.4rem',
                  background: feedback.type === 'success' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                  color: feedback.type === 'success' ? '#34d399' : '#f87171',
                  border: `1px solid ${feedback.type === 'success' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`
                }}
              >
                {feedback.type === 'success' ? <CheckCircle2 size={14} /> : <AlertCircle size={14} />}
                {feedback.message}
              </div>
            )}

            {/* Abstract */}
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', lineHeight: 1.55, marginBottom: '1.1rem' }}>
              {paper.abstract || <em style={{ color: '#94a3b8' }}>Abstract unavailable from paper source.</em>}
            </p>

            {/* Retrieval Score Breakdown Chips */}
            {(paper.bm25_score !== null || paper.tfidf_score !== null || paper.semantic_score !== null || paper.rrf_score !== null) && (
              <div
                style={{
                  display: 'flex',
                  flexWrap: 'wrap',
                  gap: '0.75rem',
                  padding: '0.65rem 0.9rem',
                  background: 'rgba(15, 23, 42, 0.55)',
                  borderRadius: '10px',
                  border: '1px solid rgba(255, 255, 255, 0.07)',
                  fontSize: '0.78rem',
                  color: '#94a3b8',
                  marginBottom: '1rem',
                  alignItems: 'center'
                }}
              >
                <span style={{ fontWeight: 700, color: '#cbd5e1', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                  <Layers size={13} color="var(--primary-teal)" /> Retrieval Scores:
                </span>
                {paper.bm25_score !== null && paper.bm25_score !== undefined && (
                  <span>BM25: <strong style={{ color: '#e2e8f0' }}>{Number(paper.bm25_score).toFixed(3)}</strong></span>
                )}
                {paper.tfidf_score !== null && paper.tfidf_score !== undefined && (
                  <span>TF-IDF: <strong style={{ color: '#e2e8f0' }}>{Number(paper.tfidf_score).toFixed(3)}</strong></span>
                )}
                {paper.semantic_score !== null && paper.semantic_score !== undefined && (
                  <span>MiniLM: <strong style={{ color: '#e2e8f0' }}>{Number(paper.semantic_score).toFixed(3)}</strong></span>
                )}
                {paper.rrf_score !== null && paper.rrf_score !== undefined && (
                  <span>RRF: <strong style={{ color: '#e2e8f0' }}>{Number(paper.rrf_score).toFixed(4)}</strong></span>
                )}
              </div>
            )}

            {/* Metadata Footer: Authors, Year, Citations, DOI */}
            <div
              style={{
                display: 'flex',
                flexWrap: 'wrap',
                gap: '1.25rem',
                fontSize: '0.82rem',
                color: 'var(--text-dim)',
                borderTop: '1px solid var(--border-subtle)',
                paddingTop: '0.85rem'
              }}
            >
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                <User size={14} />
                {paper.authors && paper.authors.length > 0 ? paper.authors.slice(0, 3).join(', ') + (paper.authors.length > 3 ? ' et al.' : '') : 'Unknown Authors'}
              </span>

              <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                <Calendar size={14} /> {paper.publication_year || 'N/A'}
              </span>

              {paper.citation_count !== null && paper.citation_count !== undefined && (
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                  <Quote size={14} /> {paper.citation_count.toLocaleString()} citations
                </span>
              )}

              {paper.doi && (
                <span style={{ color: '#94a3b8' }}>
                  DOI: <a href={`https://doi.org/${paper.doi}`} target="_blank" rel="noreferrer" style={{ color: '#93c5fd', textDecoration: 'underline' }}>{paper.doi}</a>
                </span>
              )}
            </div>

            {/* Keywords */}
            {paper.keywords && paper.keywords.length > 0 && (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginTop: '0.75rem' }}>
                {paper.keywords.map((kw, kIdx) => (
                  <span key={kIdx} className="tag">{kw}</span>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
