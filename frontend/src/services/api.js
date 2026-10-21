const API_BASE = 'http://127.0.0.1:8000/api';

function triggerBlobDownload(blob, filename) {
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.style.display = 'none';
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  }, 200);
}

export async function analyzeIdea(payload) {
  const response = await fetch(`${API_BASE}/ideas/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || 'Analysis request failed.');
  }
  return await response.json();
}

export async function checkHealth() {
  const response = await fetch(`${API_BASE}/health`);
  return await response.json();
}

export async function downloadReportPdf(analysisId, topic = '') {
  if (!analysisId) {
    throw new Error('Analysis ID is required to generate the complete research report.');
  }

  const response = await fetch(`${API_BASE}/reports/${encodeURIComponent(analysisId)}/pdf`, {
    method: 'GET'
  });

  if (!response.ok) {
    let errorMsg = 'Failed to generate report PDF.';
    try {
      const errorJson = await response.json();
      errorMsg = errorJson.detail || errorMsg;
    } catch (_) {
      const text = await response.text();
      if (text) errorMsg = text;
    }
    throw new Error(errorMsg);
  }

  const blob = await response.blob();
  const safeTopic = topic ? topic.replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 30) : 'Report';
  const filename = `Research_Report_${safeTopic}_${analysisId}.pdf`;
  triggerBlobDownload(blob, filename);
  return { success: true, filename };
}

export async function downloadPaperPdf(pdfUrl, paperTitle = 'research_paper') {
  if (!pdfUrl) {
    throw new Error('No PDF URL available for this research paper.');
  }

  const endpoint = `${API_BASE}/papers/download?url=${encodeURIComponent(pdfUrl)}&title=${encodeURIComponent(paperTitle)}`;
  const response = await fetch(endpoint, {
    method: 'GET'
  });

  if (!response.ok) {
    let errorMsg = 'Direct PDF download is restricted by publisher.';
    try {
      const errorJson = await response.json();
      errorMsg = errorJson.detail || errorMsg;
    } catch (_) {
      const text = await response.text();
      if (text) errorMsg = text;
    }
    throw new Error(errorMsg);
  }

  const blob = await response.blob();
  const safeTitle = paperTitle.replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 60);
  const filename = `${safeTitle}.pdf`;
  triggerBlobDownload(blob, filename);
  return { success: true, filename };
}
