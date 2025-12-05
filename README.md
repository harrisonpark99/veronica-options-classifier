# VERONICA

**V**irtual **E**nhanced **R**esearch & **O**perations **N**etwork for **I**nstitutional **C**rypto **A**nalytics

## Overview

VERONICA is a multi-page Streamlit application for institutional crypto analytics, featuring:

- **Authentication Gate**: Password-protected access to all features
- **Option Classifier**: CSV-based option deal classification and aggregation
- **Xunke Support**: OKX API integration for real-time and historical price data

## Project Structure

```
veronica-options-classifier/
├── app.py                    # Main entry point (Login page)
├── pages/
│   ├── 1_Option_Classifier.py    # Option deal classification
│   └── 2_Xunke_Support.py        # OKX price integration
├── utils/
│   ├── __init__.py
│   ├── auth.py               # Authentication utilities
│   ├── common.py             # Common utilities and config
│   └── okx_api.py            # OKX API functions
├── requirements.txt
└── README.md
```

## Features

### Option Classifier
- CSV option deal classification
- Automatic product type inference (Put, Call, Bonus Coupon, Sharkfin)
- Expiry-based filtering (Non-expired, M+1, M+2, M+3)
- Token Amount aggregation by Product Type
- CSV download support

### Xunke Support
- OKX real-time spot price fetching
- Historical daily close price retrieval (1D candles with pagination)
- Qty * Month (USD) calculation
- Counterparty-based aggregation
- API debug tools

## Setup

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variable or create `.streamlit/secrets.toml`:
```toml
APP_PASSWORD = "your_password_here"
```

3. Run the app:
```bash
streamlit run app.py
```

### Streamlit Cloud Deployment

1. Connect your GitHub repository to Streamlit Cloud
2. Set the main file path to `app.py`
3. Add `APP_PASSWORD` to Secrets in Streamlit Cloud settings

## Requirements

- Python 3.9+
- Streamlit 1.50.0+
- See `requirements.txt` for full dependencies

## Security

- Password authentication required for all pages
- Session-based authentication state
- No sensitive data stored locally

## License

Internal use only.
