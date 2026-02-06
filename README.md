# Bank Fraud Detection & Risk Scoring Dashboard

## Professional Big Data Analytics Platform for Financial Institutions

### Overview
This is a comprehensive, enterprise-grade fraud detection dashboard built for banking consultations and demonstrations. The system simulates real-time big data analysis with professional visualizations and interactive risk assessment capabilities.

---

## Features

### Tab 1: Executive Dashboard (Analyst View)
- **Real-time KPI Metrics**
  - Total transactions processed
  - Fraud detection rate percentage
  - Total blocked fraudulent amount
  - Overall transaction volume
  
- **Advanced Visualizations**
  - Time-series analysis showing normal vs. fraudulent transactions over 24 hours
  - Device-based fraud distribution (Mobile, Desktop, Tablet)
  - Transaction amount distribution histograms
  - Geographic risk analysis by location
  - Payment method fraud comparison
  - Merchant category funnel analysis

### Tab 2: Live Transaction Monitoring (Engineer View)
- **Real-time Transaction Stream Simulator**
  - Live feed of transaction processing
  - Individual transaction details (User ID, Amount, Location, Device)
  - Processing time metrics
  - Color-coded fraud alerts:
    - RED with "BLOCKED" status for fraudulent transactions
    - GREEN with "APPROVED" status for safe transactions
  
- **Transaction Data Grid**
  - Detailed transaction history table
  - Sortable and filterable columns

### Tab 3: Risk Scoring Framework (Consultant View)
- **Interactive Risk Calculator**
  - Adjustable parameters:
    - Transaction amount slider (0-10,000 USD)
    - Distance from home slider (0-500 km)
    - Typing speed slider (50-300 ms/char)
    - Device type selection
    - Payment method selection
    - Time of day selection
  
- **Dynamic Risk Assessment**
  - Real-time risk score gauge (0-100)
  - Color-coded risk levels:
    - GREEN (0-39): Low Risk - Approve Transaction
    - YELLOW (40-69): Elevated Risk - Require Additional Verification
    - RED (70-100): Critical Risk - Block Immediately
  
- **Risk Factor Breakdown**
  - Detailed list of identified risk factors
  - Weighted scoring system explanation
  - Actionable recommendations

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Install Required Libraries
```bash
pip install streamlit pandas numpy plotly
```

### Step 2: Download the Dashboard
Save the `fraud_detection_dashboard.py` file to your local directory.

---

## Usage

### Running the Dashboard

1. Open your terminal or command prompt
2. Navigate to the directory containing the file:
   ```bash
   cd path/to/your/directory
   ```

3. Run the Streamlit application:
   ```bash
   streamlit run fraud_detection_dashboard.py
   ```

4. The dashboard will automatically open in your default web browser at `http://localhost:8501`

### Navigation

- Use the **three tabs** at the top to switch between different views
- Use the **sidebar** to view system status and refresh data
- Click the **"START SIMULATION"** button in the Live Monitoring tab to watch real-time transaction processing
- Adjust **sliders and selectors** in the Risk Scoring tab to calculate custom risk scores

---

## Technical Architecture

### Data Generation
- Generates 1,000 realistic synthetic transactions
- Includes temporal patterns over 24-hour period
- Simulates multiple cities, payment methods, and device types
- Uses statistical distributions for realistic transaction amounts

### Fraud Detection Logic
The system uses a multi-factor scoring algorithm:
- **Amount Analysis** (30 points): Flags high-value transactions
- **Geographic Analysis** (25 points): Identifies unusual locations and international transactions
- **Distance Analysis** (20 points): Detects transactions far from home base
- **Behavioral Analysis** (15 points): Identifies bot-like typing patterns
- **Method Analysis** (10 points): Flags high-risk payment methods

Transactions scoring above 60 points are flagged as fraudulent.

### Visualization Technologies
- **Plotly Express & Graph Objects**: Interactive charts with dark theme
- **Custom CSS**: Professional styling with gradient backgrounds
- **Responsive Design**: Adapts to different screen sizes
- **Animation Effects**: Pulsing alerts for fraud detection

---

## Customization Options

### Adjusting Data Volume
Change the number of generated transactions:
```python
st.session_state.transaction_data = generate_transaction_data(2000)  # Change 1000 to desired number
```

### Modifying Fraud Detection Threshold
Adjust the fraud score threshold:
```python
fraud_indicators.append(fraud_score > 60)  # Change 60 to desired threshold
```

### Adding New Risk Factors
Extend the risk scoring in Tab 3 by adding new parameters to the calculation section.

---

## Performance Considerations

- **Data Refresh**: Use the sidebar "Refresh Data" button to generate new dataset
- **Simulation Speed**: Adjust the `time.sleep(0.5)` value in Tab 2 to control stream speed
- **Browser Performance**: For best results, use Chrome or Firefox with hardware acceleration enabled

---

## Use Cases

1. **Client Demonstrations**: Showcase fraud detection capabilities to banking clients
2. **Stakeholder Presentations**: Visualize fraud patterns and risk metrics for executives
3. **Training Sessions**: Educate fraud analysts on pattern recognition
4. **Proof of Concept**: Demonstrate real-time analytics architecture
5. **Risk Assessment Workshops**: Interactive sessions with risk officers

---

## Color Coding System

- **Blue (#00d4ff)**: Primary accent, headers, highlights
- **Green (#00c851)**: Safe transactions, low risk, approved status
- **Yellow (#ffbb33)**: Medium risk, warnings
- **Red (#ff4444)**: Fraud alerts, critical risk, blocked transactions
- **Dark Background (#0e1117)**: Professional dark theme for reduced eye strain

---

## Support & Troubleshooting

### Common Issues

**Dashboard doesn't load**
- Ensure all required packages are installed
- Check Python version compatibility (3.8+)

**Charts not displaying**
- Clear browser cache
- Try a different browser
- Verify Plotly installation: `pip install --upgrade plotly`

**Slow performance**
- Reduce data volume in the generation function
- Close unnecessary browser tabs
- Increase simulation delay in Tab 2

---

## Future Enhancements

- Machine learning model integration
- Real database connectivity
- Export functionality for reports
- Email alerting system
- Multi-user authentication
- Historical trend analysis
- Configurable alert thresholds
- Custom rule engine

---

## License
This is a demonstration project for consulting purposes.

## Contact
For customization requests or technical support, please contact your development team.

---

**Built with Streamlit, Plotly, and Python**
