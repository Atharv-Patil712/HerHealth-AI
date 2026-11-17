# HerHealth AI

A thoughtfully designed, trustworthy women's health education, cycle tracking, and wellness platform centered on menstrual health, symptom awareness, wellness reflections, and personalized evidence-based education.

**Medical Safety & Privacy Notice:** HerHealth AI is a wellness and educational application, not a medical device. It does not provide medical diagnoses, treatment plans, or clinical predictions. Menstrual cycle estimates are approximate and should not be used as contraception. The image-assisted anemia screening interface is an educational prototype with strict medical disclaimers. All personal health data (cycle entries, journal notes, bookmarks) is stored locally in your browser to respect user privacy.

---

## Key Features

1. **Menstrual Cycle Tracking**
   - Interactive monthly calendar with previous/next month navigation.
   - Comprehensive daily logging: period start/end dates, flow intensity (spotting, light, medium, heavy), cramp severity (mild, moderate, severe), symptoms (bloating, headache, fatigue, acne, backache, nausea), mood, energy levels, and private notes.
   - Intelligent cycle calculations based solely on user-recorded periods (cycle length, period duration).
   - Next period, fertile window, and estimated ovulation projections with clear uncertainty notices.
   - Edit, delete, and persistent browser storage.

2. **Women's Health Education Library**
   - 12 comprehensive, evidence-based articles covering menstrual cycle basics, period pain, PCOS, endometriosis, PMS/PMDD, anemia & iron deficiency, iron-rich nutrition, vaginal health, menstrual hygiene, cycle irregularities, when to seek medical care, and preparing for clinical appointments.
   - Fast keyword search across titles and article content.
   - Category filtering (All, Menstrual Health, Conditions, Nutrition, Care Guidance).
   - Bookmark management with local persistence.
   - Citations and reference links to authoritative health authorities (WHO, NHS, ACOG, CDC).

3. **Wellness Journal**
   - Daily logs for mood, energy, sleep quality, physical symptoms, and free-text reflections.
   - Overview patterns and wellness insights calculated across logged days.
   - Secure local storage with complete edit and delete control.

4. **Image-Assisted Anemia Screening (Prototype)**
   - Clear informed consent gate and medical safety disclaimers (a blood test is the only definitive test for anemia).
   - Dual-image capture guidance for inner lower eyelid (palpebral conjunctiva) and fingernails/nail beds.
   - Client-side validation for file types and file sizes.

5. **Global Multilingual Support**
   - Available across the entire application in 4 languages:
     - **English** (en)
     - **हिन्दी / Hindi** (hi)
     - **मराठी / Marathi** (mr)
     - **ಕನ್ನಡ / Kannada** (kn)
   - Proper typography with Google Fonts (*Noto Sans Devanagari* and *Noto Sans Kannada*).
   - Instant language switching persisted in `localStorage` without raw translation keys or UI flickers.

---

## Visual Identity & Design System

An elegant, warm, feminine visual design:
- **Backgrounds:** Warm blush white (`#FFF8FA`), warm cream (`#FFF3E8`), white (`#FFFFFF`)
- **Accents:** Baby pink (`#F8DCE6`), pale rose (`#FCEBF1`), primary pink (`#E9A1BA`), deep plum/rose (`#8D435F`)
- **Typography:** Dark charcoal (`#382C32`) for high contrast, serif headings (`font-serif`) for elegance, legible sans-serif for UI controls
- **Borders & Shadows:** Soft neutral borders (`#F0DDE3`) and subtle elevated shadows

---

## Project Structure

```
HerHealth/
├── README.md
├── frontend/                     # React 18 + TypeScript + Vite + Tailwind CSS
│   ├── src/
│   │   ├── components/layout/    # Header, Footer, PageShell, ComingSoon
│   │   ├── components/ui/        # Button, Card, Link, Input, Badge
│   │   ├── lib/                  # api.ts, i18n.ts, cycleStorage.ts, journalStorage.ts
│   │   ├── locales/              # en.json, hi.json, mr.json, kn.json
│   │   ├── pages/                # Home, Dashboard, CycleTracking, Education, Journal, AnemiaScreening, NotFound
│   │   ├── App.tsx               # Client-side router
│   │   └── main.tsx              # Application entry point
│   ├── public/locales/           # Static locale fallbacks
│   ├── index.html
│   ├── package.json
│   ├── tailwind.config.js
│   └── vite.config.ts
└── backend/                      # FastAPI + SQLite + Python 3.12+
    ├── app/
    │   ├── main.py               # FastAPI app definition & CORS
    │   ├── config.py             # Settings
    │   ├── database.py           # SQLite engine & session maker
    │   ├── models.py             # SQLAlchemy models
    │   ├── schemas.py            # Pydantic validation schemas
    │   └── routers/              # health, dashboard, education, cycle, screenings
    ├── tests/                    # Pytest test suite
    └── requirements.txt
```

---

## Getting Started

### 1. Backend (FastAPI)

```bash
cd backend
# Create virtual environment if not already created
python -m venv .venv

# Activate virtual environment
# Windows CMD:
.venv\Scripts\activate.bat
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
- API Docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/api/health`

### 2. Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```
- Web Application: `http://localhost:5173`
- Vite automatically proxies `/api` calls to `http://localhost:8000`.

---

## Running Verification & Tests

### Frontend Tests (Vitest + React Testing Library)
```bash
cd frontend
npm test -- --run
```
*Current test suite: 5 test files, 20 passing unit & integration tests.*

### Frontend Production Build
```bash
cd frontend
npm run build
```
*Executes TypeScript type checking (`tsc --noEmit`) and generates the production bundle via Vite with 0 errors.*

### Backend Tests (Pytest)
```bash
cd backend
.venv\Scripts\pytest.exe
```
*Current test suite: 12 passing tests.*

Some Important Points :->

 HerHealth AI — Models, Datasets & Engines by Task

• For Inner Lower Eyelid (Conjunctiva) Detection & Anemia Screening:
	• Model: HerHealthUnifiedModel (Custom Fused PyTorch Pipeline saved as herhealth_unified.pt)
	• How it works: Combines Fused YOLOv11n-seg (detects and crops the inner lower eyelid region of interest) with Fused EfficientNet-B0 (predicts anemia probability and estimates hemoglobin in g/dL).
	• Backend Endpoint: POST /api/screenings/analyze


• For Fingernail Bed Pallor & Anemia Screening:
	• Model: JetX-GT/nail-anemia-detector (Hugging Face Pretrained Model: mlp_model.joblib + feature_scaler.joblib)
	• How it works: Extracts 24 color, redness, brightness, and pallor features (extract_features) from the fingernail bed and runs a scikit-learn MLPClassifier tuned for high-sensitivity screening (90% recall).
	• Backend Endpoint: POST /api/screenings/analyze-nail


• For Women's Health Q&A & Medical Knowledge Base:
	• Dataset & Search Engine: altaidevorg/women-health-mini (10,348 conversational medical Q&A pairs) indexed in SQLite FTS5 (women_health_qa virtual table)
	• How it works: Uses BM25 full-text relevance ranking with stopword filtering to retrieve verified answers on menstruation, anemia, PCOS, and reproductive care in milliseconds.
	• Backend Endpoint: GET /api/education/ask


• For Hands-Free Voice Dictation (Multi-Language):
	• Engine: Browser Web Speech API (SpeechRecognition / webkitSpeechRecognition) wrapped in VoiceMicButton.tsx
	• How it works: Captures spoken user queries in real time and transcribes them into text across supported languages (en-IN, hi-IN, mr-IN, ta-IN, te-IN, bn-IN).


• For Menstrual Cycle, Fertile Window & Ovulation Prediction:
	• Engine: Rule-Based Clinical Cycle Calculator (cycleStorage.ts) backed by SQLite (cycle_entries)
	• How it works: Computes average cycle length from logged period start dates and projects the next predicted period, ovulation day, and fertile window.
	• Backend Endpoint: POST /api/cycle-entries & GET /api/cycle-entries

   
• For Permanent User History & Dashboard Analytics:
	• Database: SQLite (backend/data/herhealth.db) via Python sqlite3 (database.py)
	• How it works: Stores all screening predictions (anemia_screenings), cycle logs (cycle_entries), and Q&A records (women_health_qa) permanently on disk.
	• Backend Endpoint: GET /api/screenings/history & GET /api/dashboard
