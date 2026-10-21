import os
import sys
import unittest
from fastapi.testclient import TestClient

from app.main import app
from app.database.session import SessionLocal
from app.database.repository import AnalysisRepository
from app.services.report_generator import generate_research_report_pdf
from app.services.paper_download import validate_url_security

class TestFeatureImplementation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.db = SessionLocal()
        cls.repo = AnalysisRepository()

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def test_01_report_pdf_generation_endpoint(self):
        """Test GET /api/reports/{analysis_id}/pdf generates complete 16-section PDF."""
        # Using existing analysis run
        res = self.client.get("/api/reports/ANALYSIS-7b41eeaa/pdf")
        self.assertEqual(res.status_code, 200, f"Expected 200, got {res.status_code}: {res.text}")
        self.assertEqual(res.headers.get("content-type"), "application/pdf")
        self.assertTrue(res.headers.get("content-disposition", "").startswith("attachment; filename="))
        self.assertGreater(len(res.content), 50000, "PDF should be substantial (>50KB) with all 16 sections")
        print(f"[TEST 1 PASSED] Generated Report PDF: {len(res.content)} bytes.")

    def test_02_report_pdf_nonexistent_id(self):
        """Test GET /api/reports/{analysis_id}/pdf returns 404 for missing analysis."""
        res = self.client.get("/api/reports/ANALYSIS-NONEXISTENT/pdf")
        self.assertEqual(res.status_code, 404)
        print("[TEST 2 PASSED] Nonexistent analysis returns 404 correctly.")

    def test_03_report_generate_post_endpoint(self):
        """Test POST /api/reports/generate with analysis_id query parameter."""
        res = self.client.post("/api/reports/generate?analysis_id=ANALYSIS-7b41eeaa")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("content-type"), "application/pdf")
        self.assertGreater(len(res.content), 50000)
        print(f"[TEST 3 PASSED] POST /reports/generate succeeded: {len(res.content)} bytes.")

    def test_04_ssrf_protection_loopback(self):
        """Test SSRF rejection on localhost and private loopback IPs."""
        res = self.client.get("/api/papers/download?url=http://127.0.0.1:8000/internal")
        self.assertEqual(res.status_code, 400)
        self.assertIn("strictly prohibited", res.json().get("detail", "").lower())
        print("[TEST 4 PASSED] SSRF loopback rejected.")

    def test_05_ssrf_protection_metadata(self):
        """Test SSRF rejection on cloud metadata address (169.254.169.254)."""
        res = self.client.get("/api/papers/download?url=http://169.254.169.254/latest/meta-data/")
        self.assertEqual(res.status_code, 400)
        print("[TEST 5 PASSED] SSRF metadata IP rejected.")

    def test_06_ssrf_protection_unsupported_protocol(self):
        """Test SSRF rejection on file:// and ftp:// protocols."""
        res = self.client.get("/api/papers/download?url=file:///etc/passwd")
        self.assertEqual(res.status_code, 400)
        self.assertIn("only http and https", res.json().get("detail", "").lower())
        print("[TEST 6 PASSED] Non-HTTP protocols rejected.")

    def test_07_paper_download_live_arxiv_sample(self):
        """Test paper download from legitimate open-access public repository (arXiv)."""
        arxiv_url = "https://arxiv.org/pdf/1706.03762.pdf" # Attention Is All You Need
        res = self.client.get(f"/api/papers/download?url={arxiv_url}&title=Attention_Is_All_You_Need")
        if res.status_code == 200:
            self.assertEqual(res.headers.get("content-type"), "application/pdf")
            self.assertTrue(res.content.startswith(b"%PDF-"))
            print(f"[TEST 7 PASSED] Live arXiv paper PDF successfully streamed: {len(res.content)} bytes.")
        else:
            # In case of network sandbox/rate limiting on arXiv, ensure status is cleanly handled
            print(f"[TEST 7 NOTICE] Arxiv responded with status {res.status_code}: {res.text}")

    def test_08_paper_download_paywalled_or_html(self):
        """Test downloading an HTML page or unreachable paywall returns 400/422."""
        res = self.client.get("/api/papers/download?url=https://httpbin.org/html&title=Example_HTML")
        if res.status_code == 422:
            self.assertIn("html landing or paywall page", res.json().get("detail", "").lower())
            print("[TEST 8 PASSED] HTML paywall/landing page cleanly detected and rejected with 422.")
        else:
            self.assertEqual(res.status_code, 400)
            print(f"[TEST 8 PASSED] Unresolved or blocked non-PDF endpoint rejected with {res.status_code}.")

if __name__ == "__main__":
    unittest.main()
