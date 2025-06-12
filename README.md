# Future of Canada  
Wisdom Lantern  VanAI Hackathon Round 2 Submission  
_Exploring Canadian identity, values, and aspirations through data, code, and storytelling_

---

## What Is This Repository?

This project analyzes the **Canadian Identity Survey (2024)**—a national questionnaire that asked 1 000+ Canadians about their sense of belonging, top values, perceived social divisions, and hopes or fears for the country’s future.  
We combine quantitative statistics, natural-language processing, and data-driven storytelling (videos + interactive web viz) to reveal:

* How strongly Canadians feel connected to the nation  
* Which values they want Canada to champion next  
* Where they see the deepest fault lines—and how we might bridge them  

The result is an executive dashboard, rich visual assets, and an HTML narrative submitted to the VanAI “Future of Canada” hackathon.

---

## Repository Structure

### Python / Shell Scripts
| Script | Purpose |
|--------|---------|
| `scripts/canadian-identity-analysis.py` | End-to-end analysis pipeline.  • Loads raw CSVs • Cleans & structures responses • Performs theme extraction/NLP • Builds visualizations • Writes processed JSON + an executive dashboard text file. |
| `scripts/transcribe_videos.sh` | Batch-cleans survey response videos with `ffmpeg` and generates transcripts with **insanely-fast-whisper** (GPU/MPS). |

### Original Data
* `original_data/Hackathon Round 2_*.csv` – numeric survey results, dictionaries, extra demographics  
* `original_data/The Canadian Identity Survey_ Data Storytelling Hackathon Round 2.pdf` – official brief & questions

### Processed Data
* `processed_data/comprehensive_analysis.json` – master analysis output consumed by the web viz  
* `processed_data/detailed_analysis/` – thematic JSON break-outs (values, divisions, aspirations, etc.)  
* `processed_data/visualizations/*.png` – Matplotlib/Seaborn plots used in the write-up  
* `processed_data/executive_dashboard.txt` – plain-text “at-a-glance” dashboard

### Hackathon Entry
hackathon_entry/
├─ index.html # Interactive story / insight explorer
├─ future-of-canada.mp4 # <60 s project explainer
├─ canadian_identity_hero.png
├─ Qvideo.mp4 # Highlight reels from respondents


---

## Key Research Questions

1. How connected do Canadians feel to their national identity today?  
2. Which _values & priorities_ do they believe should guide Canada’s future?  
3. What social divisions worry Canadians the most?  
4. How do connection strength and priorities vary by province, generation, or urban/rural status?  
5. What concrete actions do respondents suggest to build unity and prosperity?

---

## Analysis Approach

📊 **Data Collection & Processing**  
• 14-question survey of 1 001 Canadians across 11 provinces / territories  
• Mixed-methods: multiple-choice, open-ended text, video responses, demographics  
• Automated Whisper transcription of videos with manual verification → thematic personas  

🧠 **AI-Powered Analysis**  
• Quantitative: pandas / NumPy stats, response-completeness metrics, demographic pattern recognition  
• Qualitative: NLP sentiment analysis, keyword & theme extraction, paradox-detection models surfacing identity tensions  
• Autonomous Python agents orchestrate pattern recognition and insight generation  

🎙️ **Cultural Representation & Voice Synthesis**  
• Hero video narrated via Brody’s First Nations voice model (full consent) for authentic Indigenous perspective  
• Ethical integration safeguards respectful framing of findings  

🔄 **End-to-End Pipeline**  
Survey → Video Transcription → AI Analysis → Pattern Recognition → Voice Synthesis → Interactive Dashboard
---

## Getting Started

1. **Install dependencies (Python 3.9+)**  
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tqdm
   # plus ffmpeg + insanely-fast-whisper if transcribing videos
   ```

2. **Run the core analysis**  
   ```bash
   python scripts/canadian-identity-analysis.py \
          --data-dir original_data \
          --output-dir processed_data
   ```
   This regenerates all JSON outputs, plots, and the executive dashboard.

3. **(Optional) Transcribe respondent videos**  
   ```bash
   bash scripts/transcribe_videos.sh
   ```

4. **Explore the interactive story**  
   Open `hackathon_entry/index.html` in your browser.

---

## Results Snapshot

* **Total Responses:** 1 001  
* **Average Response Completeness:** 49 %  
* **Top Values:** Freedom & Autonomy (62 %), Equitable Systems (47 %), Tech Innovation (40 %)  
* **Critical Concern:** Political Polarization (61 % flag as “high”)  

See `processed_data/executive_dashboard.txt` for the full dashboard.

---

## License

This project is licensed under the license included in the LICENSE file.

---

## Acknowledgements

* VanAI “Future of Canada” Hackathon for the data set and challenge  
* OpenAI Whisper + insanely-fast-whisper for rapid video transcription  
* Matplotlib, Seaborn, and D3.js for visualization tooling