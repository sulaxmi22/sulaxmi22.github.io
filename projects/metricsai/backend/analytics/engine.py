"""
MetricsAI — Data Engine
Generates realistic business metrics for the dashboard.
In production, this would connect to PostgreSQL + Redis.
"""

import random
import math
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict


@dataclass
class MetricsSnapshot:
    """A point-in-time snapshot of business metrics."""
    timestamp: str
    revenue: float
    active_users: int
    api_calls: int
    error_rate: float
    avg_response_time_ms: float
    cpu_usage: float
    memory_usage: float
    requests_per_second: float
    conversion_rate: float


class DataEngine:
    """
    Generates realistic time-series business metrics.
    Simulates real patterns: daily cycles, trends, anomalies.
    """

    def __init__(self):
        self.tick = 0
        self.base_revenue = 15000
        self.base_users = 1200
        self.history: list[dict] = []
        self._generate_history(60)  # Pre-generate 60 data points

    def _generate_history(self, count: int):
        """Generate historical data points."""
        now = datetime.now()
        for i in range(count, 0, -1):
            ts = now - timedelta(minutes=i)
            snapshot = self._generate_snapshot(ts, self.tick)
            self.history.append(asdict(snapshot))
            self.tick += 1

    def _generate_snapshot(self, ts: datetime, tick: int) -> MetricsSnapshot:
        """Generate a single metrics snapshot with realistic patterns."""
        # Time-of-day factor (simulate daily cycles)
        hour = ts.hour
        daily_factor = 0.5 + 0.5 * math.sin((hour - 6) * math.pi / 12)

        # Trend (slight upward over time)
        trend = 1 + tick * 0.001

        # Random noise
        noise = random.gauss(1, 0.05)

        # Anomaly injection (rare)
        anomaly = 1.0
        if random.random() < 0.02:
            anomaly = random.choice([0.3, 2.5])  # Drop or spike

        revenue = self.base_revenue * daily_factor * trend * noise * anomaly
        users = int(self.base_users * daily_factor * trend * noise)
        api_calls = int(users * random.uniform(3, 8))
        error_rate = max(0, random.gauss(2.1, 0.8) * (1 / anomaly if anomaly < 1 else 1))
        response_time = max(50, random.gauss(180, 40) * (1.5 if anomaly > 2 else 1))
        cpu = min(95, max(10, random.gauss(45, 15) * daily_factor))
        memory = min(90, max(30, random.gauss(55, 10)))
        rps = max(1, api_calls / 60 + random.gauss(0, 2))
        conv_rate = max(0.5, min(12, random.gauss(4.2, 1.5)))

        return MetricsSnapshot(
            timestamp=ts.strftime("%H:%M:%S"),
            revenue=round(revenue, 2),
            active_users=max(0, users),
            api_calls=max(0, api_calls),
            error_rate=round(min(100, error_rate), 2),
            avg_response_time_ms=round(response_time, 1),
            cpu_usage=round(cpu, 1),
            memory_usage=round(memory, 1),
            requests_per_second=round(rps, 1),
            conversion_rate=round(conv_rate, 2),
        )

    def get_latest(self) -> dict:
        """Generate and return the latest metrics snapshot."""
        snapshot = self._generate_snapshot(datetime.now(), self.tick)
        data = asdict(snapshot)
        self.history.append(data)
        self.tick += 1

        # Keep only last 120 points
        if len(self.history) > 120:
            self.history = self.history[-120:]

        return data

    def get_history(self, points: int = 60) -> list[dict]:
        """Get historical data points."""
        return self.history[-points:]

    def get_summary(self) -> dict:
        """Get aggregated summary of recent metrics."""
        recent = self.history[-30:] if len(self.history) >= 30 else self.history
        if not recent:
            return {}

        return {
            "avg_revenue": round(sum(d["revenue"] for d in recent) / len(recent), 2),
            "total_revenue": round(sum(d["revenue"] for d in recent), 2),
            "avg_users": int(sum(d["active_users"] for d in recent) / len(recent)),
            "peak_users": max(d["active_users"] for d in recent),
            "avg_error_rate": round(sum(d["error_rate"] for d in recent) / len(recent), 2),
            "avg_response_time": round(sum(d["avg_response_time_ms"] for d in recent) / len(recent), 1),
            "total_api_calls": sum(d["api_calls"] for d in recent),
            "data_points": len(recent),
        }
