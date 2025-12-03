# Unified CRM Dashboard Suite — Online School Analytics

**Live dashboard** → https://unified-dashboard-t2tr.onrender.com/deals

Interactive multi-page analytical application on **Plotly Dash**, combining 4 key blocks of CRM data from online schools (2023–2024):

### Includes 4 dashboards
- Deals Performance — воронки, конверсии, выручка (€978k+), Win Rate по менеджерам
- Calls Overview — 95k+ звонков, длительность, активность по часам, типы звонков
- Marketing & ROI — LTV, CPA, CLTV, ROI по каналам (Facebook Ads, Google Ads, TikTok, Organic)
- Geo Dashboard — география сделок по Германии и Европе, уровень языка студентов

### Key metrics
- Total Leads: 21,593 → Conversion Real: 3.9 % → Revenue: €978,650
- Total Calls: 95,874 (avg 164.8 сек, 76.1 % meaningful)
- Best channels: Facebook Ads, Google Ads, TikTok Ads
- Top managers by Win Rate and revenue

### Technologies
- Python 3.11, Plotly Dash, Pandas, NumPy
- Gunicorn + Render.com (production deployment)
- Responsive design, filters, navigation between dashboards

### Скриншоты
![Deals Dashboard](screenshots/deals.png)
![Calls Dashboard](screenshots/calls.png)
![Marketing & ROI](screenshots/marketing.png)
![Geo Dashboard](screenshots/geo.png)

### Локальный запуск
```bash
pip install -r requirements.txt
python app.py
