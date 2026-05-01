# MetricsAI — Real-Time AI Analytics Dashboard

<p align="center">
  <img src="https://img.shields.io/badge/FastAPI-Backend-green?style=flat-square&logo=fastapi" />
  <img src="https://img.shields.io/badge/Chart.js-4.x-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/WebSocket-Real--Time-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/OpenAI-Insights-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/Docker-Ready-blue?style=flat-square&logo=docker" />
</p>

## 🎯 Problem Statement

Business teams need real-time visibility into key metrics, but traditional dashboards are static and require manual analysis. MetricsAI combines **WebSocket-powered real-time streaming**, **interactive Chart.js visualizations**, and **LLM-powered anomaly detection** to deliver a dashboard that not only shows data but explains it.

## 🏗️ Architecture

```
┌───────────────────────────────────────────────┐
│              Browser Dashboard                 │
│   Chart.js · WebSocket · Natural Language UI   │
└────────────────────┬──────────────────────────┘
                     │ WebSocket + REST
┌────────────────────▼──────────────────────────┐
│              FastAPI Backend                    │
│     /ws/stream · /api/metrics · /api/insights  │
├────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────────────┐    │
│  │ Data Engine │  │   AI Insights Engine  │    │
│  │ (Generator  │  │   (OpenAI LLM for    │    │
│  │  + SQLite)  │  │   anomaly detection  │    │
│  └─────────────┘  │   & trend analysis)  │    │
│                    └──────────────────────┘    │
└────────────────────────────────────────────────┘
```

## ✨ Features

- **Real-time streaming** via WebSockets — metrics update every second
- **6 interactive charts** — line, bar, doughnut, area, with zoom and pan
- **AI-powered insights** — LLM analyzes data patterns, detects anomalies
- **Natural language queries** — ask questions about your data in plain English
- **Responsive dark theme** — glassmorphism cards, smooth animations
- **Sample data engine** — generates realistic business metrics for demo

## 🚀 Quick Start

```bash
git clone https://github.com/sulaxmi22/metricsai.git
cd metricsai
pip install -r requirements.txt
cp .env.example .env
python -m uvicorn backend.main:app --reload --port 8002
# Open frontend/index.html in browser
```

## 📝 License

MIT License
