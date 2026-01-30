# 🎮 Crash Game Security Testing Suite

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-Active-success.svg)](.)

**Advanced security testing and prediction system for crash-style casino games using machine learning and statistical analysis.**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Test Results Summary](#-test-results-summary)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Test Modes](#-test-modes)
- [Output Files](#-output-files)
- [Security Analysis](#-security-analysis)
- [Known Issues](#-known-issues)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [Disclaimer](#-disclaimer)

---

## 🔍 Overview

This security testing suite evaluates the randomness, predictability, and potential vulnerabilities of crash game implementations. It uses multiple machine learning models, statistical tests, and live monitoring to assess game fairness and detect exploitable patterns.

### Key Capabilities:
- **Live monitoring** via Selenium WebDriver or WebSocket
- **Historical data analysis** with 10,000+ data points
- **Multi-model prediction** using XGBoost, Random Forest, LSTM, GRU, and more
- **Statistical validation** (Chi-Square, KS-Test, Runs Test, Entropy)
- **Vulnerability scoring** based on prediction accuracy vs random baseline

---

## 📊 Test Results Summary

### Latest Test: January 21, 2026

#### Statistical Analysis
| Metric | Value |
|--------|-------|
| **Data Points** | 10,000 |
| **Mean** | 2.99x |
| **Median** | 2.38x |
| **Std Deviation** | 1.98 |
| **Range** | 1.00x - 23.98x |
| **Entropy** | 0.5897 |

#### Randomness Tests
| Test | Statistic | P-Value | Result |
|------|-----------|---------|--------|
| **Chi-Square** | 41,911.66 | 0.0000 | ⚠️ Non-uniform distribution |
| **KS-Test** | 0.0049 | 0.9677 | ✅ Passes exponential distribution |
| **Runs Test** | -1.52 | 0.1285 | ✅ Random sequence |

#### Prediction Performance
| Model | MSE | Performance |
|-------|-----|-------------|
| **Ensemble** | 0.7243 | 90.87% better than random |
| **XGBoost** | 0.7481 | Individual model |
| **Random Forest** | 0.7350 | Individual model |
| **Random Baseline** | 7.9324 | Comparison baseline |

### 🚨 **Vulnerability Score: 90.87%**

**CRITICAL FINDING**: The prediction models perform **90.87% better** than random guessing, indicating the system shows **significant predictability** and potential security vulnerabilities.

---

## ✨ Features

### 1. **Automated Live Monitoring**
- Real-time crash point detection using Selenium
- WebSocket support for direct API monitoring
- Automatic screenshot debugging
- Multi-strategy element detection (Selenium + JavaScript)

### 2. **Advanced Machine Learning**
- **11 Prediction Models**: XGBoost, LightGBM, CatBoost, Random Forest, Extra Trees, Gradient Boosting, ElasticNet, Ridge, Bidirectional LSTM, GRU, Stacking Ensemble
- Bayesian hyperparameter optimization
- Feature engineering with 60+ statistical indicators
- Weighted ensemble predictions with confidence intervals

### 3. **Statistical Analysis**
- Chi-Square test for distribution uniformity
- Kolmogorov-Smirnov test for exponential distribution
- Autocorrelation analysis (20 lags)
- Runs test for randomness
- Shannon entropy calculation
- Timing correlation analysis

### 4. **Attack Simulations**
- Seed-based prediction attempts
- Pattern recognition exploits
- Timing attack detection

### 5. **Manual Input Mode**
- Interactive crash point entry
- File import (JSON, CSV, TXT)
- Prediction accuracy comparison
- Real-time model training

---

## 🚀 Installation

### Prerequisites
- Python 3.12+
- Chrome Browser (for Selenium mode)
- 8GB+ RAM (for deep learning models)

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/crash-game-security-test.git
cd crash-game-security-test
```

### Step 2: Create Virtual Environment
```bash
python -m venv testenv
```

**Windows:**
```bash
testenv\Scripts\activate
```

**Linux/Mac:**
```bash
source testenv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install numpy scipy scikit-learn tensorflow xgboost statsmodels prophet pmdarima bayesian-optimization matplotlib pytest pandas websocket-client selenium webdriver-manager lightgbm catboost
```

### Step 4: Verify Installation
```bash
python crash_security_test.py --help
```

---

## 💻 Usage

### Mode 1: Historical Data Testing (Recommended for first run)
```bash
python crash_security_test.py --num_rounds 10000
```

### Mode 2: Live Monitoring with Selenium
```bash
python crash_security_test.py --live_mode --live_duration 600 --site_url "https://eg1xbet.com/ar/games/crash"
```

### Mode 3: Live Monitoring with Debug Screenshots
```bash
python crash_security_test.py --live_mode --debug --live_duration 400
```

### Mode 4: WebSocket Live Monitoring
```bash
python crash_security_test.py --live_mode --use_websocket --ws_url "wss://your-site.com/crash/ws" --live_duration 600
```

### Mode 5: Manual Input Predictions
```bash
python manual_predictor.py
```

---

## 🎯 Test Modes

### 1️⃣ **Historical Mode** (Default)
- Generates 10,000 simulated crash points
- Runs complete statistical analysis
- Trains all ML models
- Outputs vulnerability score
- **Duration**: ~2-3 minutes

**Example:**
```bash
python crash_security_test.py
```

### 2️⃣ **Live Selenium Mode**
- Opens Chrome browser automatically
- Monitors live crash events
- Detects crashes in real-time
- Generates predictions after 20+ crashes
- **Duration**: User-defined (default: 600s)

**Example:**
```bash
python crash_security_test.py --live_mode --live_duration 1200
```

**Features:**
- Auto-downloads ChromeDriver
- Multiple element detection strategies
- JavaScript-based fallback detection
- Screenshot debugging with `--debug` flag

### 3️⃣ **WebSocket Mode**
- Connects directly to game WebSocket
- Lower latency than Selenium
- No browser overhead
- **Requires**: Valid WebSocket URL

**Example:**
```bash
python crash_security_test.py --live_mode --use_websocket --ws_url "wss://example.com/crash"
```

### 4️⃣ **Manual Input Mode**
- Interactive crash point entry
- Load from file (JSON/CSV/TXT)
- Generates next 10 predictions
- Compare with actual results

**Example:**
```bash
python manual_predictor.py
```

**Menu Options:**
```
1. Enter crash points manually (recommended: 50+ points)
2. Load crash points from file
3. Compare predictions with actual results
4. Exit
```

---

## 📁 Output Files

After running tests, you'll find these files in your directory:

### 1. `security_test_results_enhanced.json`
Complete test results with all metrics and statistics.

**Structure:**
```json
{
    "test_date": "2026-01-21T18:09:08",
    "mode": "live",
    "data_points": 10000,
    "Statistical": { ... },
    "Predictive": { ... },
    "Attacks": { ... },
    "Additional": { ... }
}
```

### 2. `crash_live_security_test.log`
Detailed execution log with timestamps.

**Sample:**
```
2026-01-21 18:01:50,086 - INFO - Setting up Chrome WebDriver...
2026-01-21 18:02:09,494 - INFO - Waiting for page to load...
2026-01-21 18:08:57,319 - INFO - Building prediction models...
```

### 3. `future_crash_points.json`
Next 10 predicted crash points (generated in live mode after 20+ crashes).

**Structure:**
```json
{
    "timestamp": "2026-01-21T18:09:08",
    "predicted_crash_points": [2.45, 1.89, 3.12, ...],
    "total_predictions": 10
}
```

### 4. `predictions_YYYYMMDD_HHMMSS.json` (Manual mode)
Detailed predictions with confidence intervals.

### 5. `predictions_YYYYMMDD_HHMMSS.txt` (Manual mode)
Human-readable prediction summary.

### 6. Debug Files (with `--debug` flag)
- `debug_screenshot.png` - Page screenshot
- `debug_page_source.html` - HTML source for element inspection

---

## 🔐 Security Analysis

### Vulnerability Indicators

#### ⚠️ **HIGH RISK** (Score > 80%)
- **Finding**: Models achieve 90.87% improvement over random
- **Implication**: Game shows exploitable patterns
- **Recommendation**: Review RNG implementation

#### ⚠️ **MEDIUM RISK** (Score 50-80%)
- Predictable sequences detected
- Pattern-based exploitation possible
- Recommend algorithm hardening

#### ✅ **LOW RISK** (Score < 50%)
- Acceptable randomness
- Low predictability
- Secure implementation

### Current Assessment

**Vulnerability Score: 90.87%** → **HIGH RISK**

**Key Findings:**
1. **Chi-Square p-value: 0.0000** - Distribution significantly differs from expected
2. **Autocorrelation detected** at lag 11 (0.0217)
3. **Timing correlation: 0.0028** - Negligible time-based patterns
4. **Max consecutive high values: 9** - Possible streak vulnerability

### Attack Vector Analysis

| Vector | Feasibility | Mitigation |
|--------|-------------|------------|
| Seed Prediction | Low | Tested: 0 successful predictions |
| Pattern Recognition | High | 90% accuracy achieved |
| Timing Attacks | Low | Correlation: 0.0028 |
| Sequence Exploitation | Medium | Runs test passed |

---

## ⚠️ Known Issues

### Issue 1: Unicode Encoding Errors (Windows)
**Symptom:**
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2713'
```

**Cause:** Windows console can't display special characters (✓, ⚠️, 💥)

**Status:** ✅ **FIXED** in latest version
- All unicode characters replaced with ASCII equivalents
- Example: `✓` → `[OK]`, `⚠️` → `[WARNING]`, `💥` → `[CRASH]`

### Issue 2: No Live Data Collected
**Symptom:**
```
[OK] Monitoring completed. Collected 0 crashes.
[WARNING] No live data collected. Falling back to historical data...
```

**Cause:** Website element selectors changed or anti-bot detection

**Solution:**
1. Run with `--debug` flag to save screenshots
2. Inspect `debug_screenshot.png` and `debug_page_source.html`
3. Update selectors in code if needed
4. Try increasing `--live_duration` to 1200+ seconds

### Issue 3: ChromeDriver Version Mismatch
**Symptom:**
```
WebDriver version mismatch
```

**Solution:** ✅ **AUTO-FIXED**
- Script uses `webdriver-manager` for automatic ChromeDriver installation
- Handles version mismatches automatically

### Issue 4: NaN Values in Manual Predictor
**Symptom:**
```
Fatal error: Input X contains NaN
```

**Status:** ✅ **FIXED**
- All NaN values replaced with 0
- Inf values handled
- Data validation added

---

## 🛠️ Troubleshooting

### Problem: Script crashes immediately
**Check:**
```bash
python --version  # Should be 3.12+
pip list | grep tensorflow  # Verify installation
```

### Problem: Chrome won't open in live mode
**Solutions:**
1. Install Chrome browser: https://www.google.com/chrome/
2. Check firewall settings
3. Try manual ChromeDriver: https://googlechromelabs.github.io/chrome-for-testing/

### Problem: Models training very slow
**Optimization:**
- Reduce `--num_rounds` to 5000
- Use manual mode instead (faster)
- Close other applications to free RAM

### Problem: WebSocket connection timeout
**Solutions:**
1. Verify WebSocket URL is correct
2. Check network connectivity
3. Try Selenium mode as alternative

### Problem: Prediction accuracy seems wrong
**Validation:**
1. Ensure minimum 20 data points
2. Use manual mode's comparison feature
3. Check `crash_predictions.log` for errors

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### Reporting Issues
1. Check existing issues first
2. Provide full error log
3. Include Python version and OS
4. Share sample data if possible

### Pull Requests
1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Test thoroughly with both modes
4. Update documentation
5. Submit PR with clear description

### Code Standards
- Follow PEP 8 style guide
- Add comments for complex logic
- Include error handling
- Update README for new features

---

## ⚖️ Disclaimer

**IMPORTANT LEGAL NOTICE:**

This software is provided **for educational and security research purposes only**.

### Intended Use:
✅ Security testing of your **own** systems  
✅ Academic research and learning  
✅ Vulnerability assessment with **explicit permission**  
✅ RNG quality verification

### Prohibited Use:
❌ Exploiting third-party systems without authorization  
❌ Violating terms of service of online platforms  
❌ Illegal gambling or fraud  
❌ Any unlawful activities

### Legal Responsibility:
- Users are **solely responsible** for compliance with local laws
- Authors assume **no liability** for misuse
- Usage of this software does **not** constitute permission to test third-party systems
- Unauthorized access to computer systems is **illegal** in most jurisdictions

### Ethical Guidelines:
1. **Always obtain permission** before testing third-party systems
2. **Report vulnerabilities responsibly** to affected parties
3. **Do not profit** from discovered vulnerabilities without authorization
4. **Respect** terms of service and user agreements

**By using this software, you agree to use it ethically and legally.**

---

## 📞 Contact

- **Project Lead**: [Your Name]
- **Email**: your.email@example.com
- **GitHub Issues**: [Issues Page](https://github.com/yourusername/crash-game-security-test/issues)
- **Discord**: [Community Server](https://discord.gg/yourserver)

---

## 📜 License

```
MIT License

Copyright (c) 2026 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

- **scikit-learn** team for machine learning framework
- **TensorFlow** developers for deep learning capabilities
- **Selenium** project for web automation
- **XGBoost**, **LightGBM**, **CatBoost** for gradient boosting implementations
- **webdriver-manager** for automatic ChromeDriver management
- Security research community for methodology guidance

---

## 📊 Project Stats

![Lines of Code](https://img.shields.io/badge/lines%20of%20code-2500%2B-blue)
![Models](https://img.shields.io/badge/ML%20models-11-green)
![Tests](https://img.shields.io/badge/tests-automated-success)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Mac-lightgrey)

---

**Last Updated:** January 30, 2026  
**Version:** 2.0.0  
**Status:** ✅ Production Ready
