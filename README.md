# 🌊 Ocean Wave Height Estimation via Radar Signal Volatility

## 📌 Project Overview
This project focuses on estimating **Significant Wave Height (Hs)** using **High-Frequency (HF) radar time-series data** by applying **financial volatility models**. The goal is to provide an efficient, scalable alternative to traditional spectral analysis methods, which are computationally expensive and slow.

By leveraging volatility modeling techniques from **quantitative finance (GARCH family)**, the project bridges concepts from **ocean engineering and financial time-series analysis** to improve both performance and accuracy in ocean wave monitoring.

---

## 🎯 Problem Statement
Traditional radar-based wave height estimation relies on complex Doppler spectrum processing, making it unsuitable for real-time or large-scale deployment. The challenge was to:
- Reduce computation time
- Maintain reliable accuracy
- Enable scalable ocean monitoring

---

## 💡 Key Insight
Research shows that **ocean wave height is directly related to the volatility of radar backscatter signals**:
- Calm seas → low signal volatility  
- Rough seas → high signal volatility  

This insight allows wave height estimation by modeling signal variability instead of performing heavy spectral computations.

---

## 📊 Data Sources
### 1. HF Radar Data (Primary Input)
- Time-series radar backscatter data collected from coastal HF radar systems
- Used to model signal volatility

### 2. Wave Buoy Data (Ground Truth)
- Physical buoy measurements of significant wave height
- Used for validation and accuracy benchmarking (not for direct training)

---

## 🧠 Methodology
1. Preprocessed raw HF radar time-series data to remove noise and inconsistencies  
2. Modeled radar signal volatility using statistical time-series models  
3. Evaluated multiple approaches, including:
   - MIDAS Hyperbolic ARCH  
   - GARCH  
   - EGARCH  
   - GJR-GARCH  
   - LSTM (benchmark comparison)  
4. Mapped estimated volatility to significant wave height using established linear relationships  
5. Validated predictions against real buoy measurements using RMSE and correlation metrics  

---

## 🚀 Model Selection & Optimization
- **MIDAS models** showed good accuracy but were computationally impractical (40+ hours runtime)
- **GJR-GARCH and EGARCH** provided the best balance between:
  - Accuracy
  - Execution speed
  - Stability under asymmetric ocean conditions

⏱️ **Processing time reduced from ~40 hours to ~2.5 hours (≈95% improvement)**

---

## ✅ Results & Impact
- High correlation between radar-based estimates and buoy measurements
- Significant reduction in computational cost
- Scalable and operationally feasible solution for real-world ocean monitoring
- Demonstrates successful cross-domain application of financial models to ocean engineering problems

---

## 🛠️ Tech Stack & Tools
- Python
- Time-series statistical modeling
- GARCH family models
- Data analysis & benchmarking
- Scientific computing libraries

---

## 👤 Role & Contributions
- Led end-to-end project coordination and model evaluation
- Defined comparison criteria (accuracy, runtime, scalability)
- Drove optimization decisions based on performance trade-offs
- Documented results and insights for technical and non-technical stakeholders

---

## 📈 Future Enhancements
- Real-time deployment pipeline
- Multi-location radar integration
- Automated model selection
- Hybrid statistical + ML approaches

---

## 📄 References
Relevant academic papers and datasets are documented in the project references for further exploration.

---

## 🔗 Contact
For questions, discussions, or collaboration opportunities, feel free to connect.
