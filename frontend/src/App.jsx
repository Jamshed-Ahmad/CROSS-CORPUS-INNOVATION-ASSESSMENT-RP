import React, { useState, useEffect } from 'react';
import { analyzeIdea, downloadReportPdf } from './services/api';
import Navbar from './components/Navbar';
import HeroSection from './components/HeroSection';
import WhySection from './components/WhySection';
import ModulesShowcase from './components/ModulesShowcase';
import EnvironmentSection from './components/EnvironmentSection';
import Dashboard3DSpotlight from './components/Dashboard3DSpotlight';
import ConsultationModal from './components/ConsultationModal';

import IdeaForm from './components/IdeaForm';
import PaperResults from './components/PaperResults';
import PatentResults from './components/PatentResults';
import CrossCorpusGraph from './components/CrossCorpusGraph';
import Innovation3DGraph from './components/Innovation3DGraph';
import GapAnalysis from './components/GapAnalysis';
import ScoreBreakdown from './components/ScoreBreakdown';
import OpportunityCards from './components/OpportunityCards';
import EvidencePanel from './components/EvidencePanel';

import { 
  ShieldAlert, 
  Lightbulb, 
  Box, 
  BookOpen, 
  ShieldCheck, 
  Network, 
  Target, 
  FileText, 
  Database, 
  AlertTriangle,
  ArrowUpRight,
  Sparkles,
  Download,
  CheckCircle2
} from 'lucide-react';

export default function App() {
  const [theme, setTheme] = useState('dark');
  const [consultationOpen, setConsultationOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [downloadingReport, setDownloadingReport] = useState(false);
  const [reportFeedback, setReportFeedback] = useState(null);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  };

  const scrollToAssessment = () => {
    const el = document.getElementById('live-assessment');
    if (el) el.scrollIntoView({ behavior: 'smooth' });
  };

  const scrollTo3D = () => {
    if (result) {
      setActiveTab('3d-graph');
      scrollToAssessment();
    } else {
      const el = document.getElementById('3d-universe');
      if (el) el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const handleExploreModule = (actionKey) => {
    setActiveTab(actionKey);
    scrollToAssessment();
  };

  const handleAnalyze = async (payload) => {
    setLoading(true);
    setError(null);
    try {
      const data = await analyzeIdea(payload);
      setResult(data);
      setActiveTab('overview');
      // Scroll smoothly down to the results
      setTimeout(() => {
        const resultsEl = document.getElementById('analysis-results-anchor');
        if (resultsEl) resultsEl.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    } catch (err) {
      setError(err.message || 'Pipeline execution failed.');
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadReport = async () => {
    if (!result?.analysis_id) return;
    setDownloadingReport(true);
    setReportFeedback(null);
    try {
      await downloadReportPdf(result.analysis_id, result.parsed_idea?.original_text);
      setReportFeedback({
        type: 'success',
        message: 'Complete research analysis report PDF generated and downloaded successfully!'
      });
    } catch (err) {
      setReportFeedback({
        type: 'error',
        message: err.message || 'Failed to download complete research report PDF.'
      });
    } finally {
      setDownloadingReport(false);
    }
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Luxury Indoglobal Sticky Navbar */}
      <Navbar 
        theme={theme}
        toggleTheme={toggleTheme}
        onOpenConsultation={() => setConsultationOpen(true)}
        onNavigateToAssessment={scrollToAssessment}
        onNavigateTo3D={scrollTo3D}
      />

      {/* Hero Section matching template visual hierarchy */}
      <HeroSection 
        onRunAssessment={scrollToAssessment}
        onExplore3D={scrollTo3D}
      />

      {/* Value Proposition / Why Dual-Corpus */}
      <WhySection />

      {/* 3 Core System Architecture Modules (Exclusively This Project) */}
      <ModulesShowcase 
        onExploreModule={handleExploreModule}
      />

      {/* 3-Environment Ecosystem (Highlighting Relatable 3rd Environment) */}
      <EnvironmentSection />

      {/* Dedicated 3D Animated Dashboard Section */}
      <Dashboard3DSpotlight 
        onLaunch3DUniverse={scrollTo3D}
      />

      {/* Live Cross-Corpus Assessment Engine */}
      <section className="section-wrapper" id="live-assessment" style={{ borderTop: '1px solid var(--border-subtle)', paddingBottom: '2rem' }}>
        <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
          <span className="section-tag">LIVE OPERATIONAL PLATFORM</span>
          <h2 className="section-heading">Cross-Corpus Assessment Engine</h2>
          <p className="section-subtext" style={{ margin: '0 auto' }}>
            Submit a hypothesis to evaluate scientific literature grounding, patent claim boundaries, unexploited research gaps, and view the animated 3D universe.
          </p>
        </div>

        {/* Interactive Idea Submission Form */}
        <IdeaForm onSubmit={handleAnalyze} loading={loading} />

        {/* Error Alert */}
        {error && (
          <div style={{ background: 'rgba(244, 63, 94, 0.15)', border: '1px solid rgba(244, 63, 94, 0.4)', borderRadius: '12px', padding: '1.25rem', color: 'var(--accent-rose)', marginBottom: '2rem' }}>
            <strong style={{ fontSize: '1rem' }}>Analysis Pipeline Error:</strong> {error}
          </div>
        )}

        <div id="analysis-results-anchor"></div>

        {/* Active Analysis Results Display */}
        {result && (
          <div>
            {/* Parsed Idea Overview Badge */}
            <div className="glass-panel" style={{ padding: '1.5rem', marginBottom: '1.75rem', display: 'flex', flexWrap: 'wrap', gap: '1rem', alignItems: 'center', justifyContent: 'space-between', borderColor: 'var(--border-gold)' }}>
              <div>
                <span className="tag tag-gold" style={{ marginRight: '0.6rem' }}>
                  Analysis ID: {result.analysis_id}
                </span>
                <span className="tag tag-purple" style={{ marginRight: '0.6rem' }}>
                  Domain: {result.parsed_idea?.domain || 'Cross-Domain'}
                </span>
                <span className="tag" style={{ background: result.metadata?.data_mode === 'live' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(148, 163, 184, 0.2)', color: result.metadata?.data_mode === 'live' ? '#10b981' : '#cbd5e1', border: '1px solid rgba(255,255,255,0.15)' }}>
                  <Database size={13} style={{ display: 'inline', marginRight: '5px' }} />
                  {result.metadata?.data_mode === 'live' ? 'Live APIs (OpenAlex & EPO OPS)' : 'Mock Deterministic Fixtures'}
                </span>
              </div>
              <div style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>
                Target Technology: <strong style={{ color: 'var(--text-main)' }}>{result.parsed_idea?.proposed_technology || 'Proposed Concept'}</strong>
              </div>
            </div>

            {/* Warnings Alert (e.g. Rate limits, timeouts, partial fixtures) */}
            {result.warnings && result.warnings.length > 0 && (
              <div style={{ background: 'rgba(245, 158, 11, 0.15)', border: '1px solid rgba(245, 158, 11, 0.4)', borderRadius: '12px', padding: '1.25rem', color: '#fbbf24', marginBottom: '1.75rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 750, marginBottom: '0.4rem' }}>
                  <AlertTriangle size={18} /> Provider Execution Warning (Status: {result.status})
                </div>
                <ul style={{ paddingLeft: '1.4rem', margin: 0, fontSize: '0.88rem' }}>
                  {result.warnings.map((w, idx) => (
                    <li key={idx}>{w}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* RESEARCH ANALYSIS COMPLETE & DOWNLOAD COMPLETE RESEARCH REPORT BANNER */}
            <div
              className="glass-panel"
              style={{
                padding: '1.75rem 2rem',
                marginBottom: '1.75rem',
                borderRadius: '16px',
                border: '1px solid rgba(13, 148, 136, 0.45)',
                background: 'linear-gradient(135deg, rgba(13, 148, 136, 0.15) 0%, rgba(30, 58, 138, 0.22) 100%)',
                display: 'flex',
                flexWrap: 'wrap',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: '1.5rem',
                boxShadow: '0 8px 32px rgba(13, 148, 136, 0.18)'
              }}
            >
              <div>
                <div
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '0.45rem',
                    fontSize: '0.78rem',
                    fontWeight: 800,
                    letterSpacing: '0.08em',
                    color: '#2dd4bf',
                    textTransform: 'uppercase',
                    marginBottom: '0.45rem',
                    background: 'rgba(13, 148, 136, 0.25)',
                    padding: '0.3rem 0.75rem',
                    borderRadius: '6px',
                    border: '1px solid rgba(45, 212, 191, 0.35)'
                  }}
                >
                  <CheckCircle2 size={15} /> RESEARCH ANALYSIS COMPLETE
                </div>
                <h3 style={{ fontSize: '1.35rem', fontWeight: 800, color: '#ffffff', margin: '0.25rem 0 0.4rem 0' }}>
                  Complete Cross-Corpus Innovation Assessment Report
                </h3>
                <p style={{ color: '#cbd5e1', fontSize: '0.9rem', margin: 0, maxWidth: '680px', lineHeight: 1.5 }}>
                  Empirically synthesized across {result.papers?.length || 0} OpenAlex papers, {result.patents?.length || 0} EPO patents, {result.relationships?.length || 0} Siamese neural relationships, {result.gaps?.length || 0} identified research gaps, and 6-factor innovation scoring.
                </p>
              </div>

              <div>
                <button
                  className="btn-download-report-main"
                  onClick={handleDownloadReport}
                  disabled={downloadingReport}
                  title="Generate and download complete 16-section research analysis report PDF"
                >
                  <Download size={18} />
                  {downloadingReport ? 'GENERATING REPORT PDF...' : 'DOWNLOAD COMPLETE RESEARCH REPORT'}
                </button>
              </div>
            </div>

            {/* Report Download Notification */}
            {reportFeedback && (
              <div
                style={{
                  fontSize: '0.88rem',
                  padding: '0.75rem 1.25rem',
                  borderRadius: '10px',
                  marginBottom: '1.5rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  background: reportFeedback.type === 'success' ? 'rgba(16, 185, 129, 0.18)' : 'rgba(239, 68, 68, 0.18)',
                  color: reportFeedback.type === 'success' ? '#34d399' : '#f87171',
                  border: `1px solid ${reportFeedback.type === 'success' ? 'rgba(16, 185, 129, 0.35)' : 'rgba(239, 68, 68, 0.35)'}`
                }}
              >
                {reportFeedback.type === 'success' ? <CheckCircle2 size={16} /> : <AlertTriangle size={16} />}
                {reportFeedback.message}
              </div>
            )}

            {/* Results Navigation Tabs */}
            <div className="tab-bar">
              <button
                className={`tab-btn ${activeTab === 'overview' ? 'active' : ''}`}
                onClick={() => setActiveTab('overview')}
              >
                <Lightbulb size={16} style={{ display: 'inline', marginRight: '0.4rem' }} /> Overview & Recommendations
              </button>
              <button
                className={`tab-btn ${activeTab === '3d-graph' ? 'active' : ''}`}
                onClick={() => setActiveTab('3d-graph')}
              >
                <Box size={16} style={{ display: 'inline', marginRight: '0.4rem', color: 'var(--gold-primary)' }} /> 
                <strong>3D Animated Universe</strong>
              </button>
              <button
                className={`tab-btn ${activeTab === 'papers' ? 'active' : ''}`}
                onClick={() => setActiveTab('papers')}
              >
                <BookOpen size={16} style={{ display: 'inline', marginRight: '0.4rem' }} /> Research Papers ({result.papers?.length || 0})
              </button>
              <button
                className={`tab-btn ${activeTab === 'patents' ? 'active' : ''}`}
                onClick={() => setActiveTab('patents')}
              >
                <ShieldCheck size={16} style={{ display: 'inline', marginRight: '0.4rem' }} /> Patent Literature ({result.patents?.length || 0})
              </button>
              <button
                className={`tab-btn ${activeTab === 'graph' ? 'active' : ''}`}
                onClick={() => setActiveTab('graph')}
              >
                <Network size={16} style={{ display: 'inline', marginRight: '0.4rem' }} /> Cross-Corpus Matrix ({result.relationships?.length || 0})
              </button>
              <button
                className={`tab-btn ${activeTab === 'gaps' ? 'active' : ''}`}
                onClick={() => setActiveTab('gaps')}
              >
                <Target size={16} style={{ display: 'inline', marginRight: '0.4rem' }} /> Gap Analysis ({result.gaps?.length || 0})
              </button>
              <button
                className={`tab-btn ${activeTab === 'evidence' ? 'active' : ''}`}
                onClick={() => setActiveTab('evidence')}
              >
                <FileText size={16} style={{ display: 'inline', marginRight: '0.4rem' }} /> Evidence Inspector ({result.evidence?.length || 0})
              </button>
            </div>

            {/* Tab Panels */}
            {activeTab === 'overview' && (
              <div>
                {result.recommendations?.length > 0 && result.recommendations[0].assessment_factors && (
                  <ScoreBreakdown factors={result.recommendations[0].assessment_factors} />
                )}
                {/* Embedded 3D Universe in Overview */}
                <Innovation3DGraph result={result} />
                <OpportunityCards recommendations={result.recommendations || []} />
              </div>
            )}

            {activeTab === '3d-graph' && <Innovation3DGraph result={result} />}
            {activeTab === 'papers' && <PaperResults papers={result.papers || []} />}
            {activeTab === 'patents' && <PatentResults patents={result.patents || []} />}
            {activeTab === 'graph' && <CrossCorpusGraph relationships={result.relationships || []} />}
            {activeTab === 'gaps' && <GapAnalysis gaps={result.gaps || []} />}
            {activeTab === 'evidence' && <EvidencePanel evidence={result.evidence || []} />}
          </div>
        )}

        {/* Operational & Legal Disclaimer Banner */}
        <div className="disclaimer-banner">
          <ShieldAlert size={22} style={{ flexShrink: 0, marginTop: '2px', color: 'var(--gold-primary)' }} />
          <div>
            <strong>Operational & Legal Disclaimer:</strong> The system is a research and decision-support prototype. It MUST NOT claim to establish legal patentability, legal freedom to operate, automated patent filing, or guaranteed commercial success. It clearly distinguishes preliminary computational assessments from decisions requiring legal, technical, intellectual-property, or commercial expertise.
          </div>
        </div>
      </section>

      {/* Luxury Indoglobal Footer */}
      <footer style={{ borderTop: '1px solid var(--border-subtle)', background: 'rgba(7, 9, 14, 0.95)', padding: '3.5rem 1.5rem 2rem 1.5rem', marginTop: 'auto' }}>
        <div style={{ maxWidth: '1320px', margin: '0 auto', display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', gap: '2rem', marginBottom: '2.5rem' }}>
          <div 
            className="brand-logo-wrap"
            onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
            style={{ cursor: 'pointer', display: 'flex', alignItems: 'center' }}
          >
            <img 
              src="/logo.png" 
              alt="Cross-Corpus Innovation Assessment and Recommendation" 
              style={{ height: '44px', width: 'auto', objectFit: 'contain' }}
            />
          </div>

          <div style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap', fontSize: '0.9rem', color: 'var(--text-muted)' }}>
            <a href="#about" style={{ color: 'inherit', textDecoration: 'none' }}>About Framework</a>
            <a href="#why" style={{ color: 'inherit', textDecoration: 'none' }}>Why Dual-Corpus</a>
            <a href="#modules" style={{ color: 'inherit', textDecoration: 'none' }}>Core Modules</a>
            <a href="#environments" style={{ color: 'inherit', textDecoration: 'none' }}>3-Environments</a>
            <a href="#3d-universe" style={{ color: 'inherit', textDecoration: 'none' }}>3D Animations</a>
            <a href="#live-assessment" style={{ color: 'inherit', textDecoration: 'none' }}>Live Assessment</a>
          </div>

          <button 
            onClick={() => setConsultationOpen(true)}
            className="btn-gold"
            style={{ padding: '0.65rem 1.5rem', fontSize: '0.82rem' }}
          >
            SCHEDULE CONSULTATION <ArrowUpRight size={15} />
          </button>
        </div>

        <div style={{ maxWidth: '1320px', margin: '0 auto', textAlign: 'center', fontSize: '0.8rem', color: 'var(--text-dim)', borderTop: '1px solid var(--border-subtle)', paddingTop: '1.5rem' }}>
          © 2026 An Evidence-Grounded AI Framework for Cross-Corpus Innovation Assessment and Recommendation Using Research Papers and Patents. All rights reserved.
        </div>
      </footer>

      {/* Consultation Modal */}
      <ConsultationModal 
        isOpen={consultationOpen} 
        onClose={() => setConsultationOpen(false)} 
      />
    </div>
  );
}
