import pytest
import asyncio
from unittest.mock import MagicMock
import gptscan

# Helper to verify time progression in tests
class TimeMachine:
    def __init__(self, start_time=100.0):
        self.current_time = start_time

    def time(self):
        return self.current_time

    def advance(self, seconds):
        self.current_time += seconds

def test_acquire_allows_immediate_access(monkeypatch):
    async def run():
        tm = TimeMachine()
        monkeypatch.setattr(gptscan.time, 'monotonic', tm.time)
        mock_sleep = MagicMock()
        monkeypatch.setattr(gptscan.asyncio, 'sleep', mock_sleep)

        limiter = gptscan.AsyncRateLimiter(rate_per_minute=5)

        # Action: Acquire 5 times
        for _ in range(5):
            await limiter.acquire()
            tm.advance(0.1) # Simulate tiny processing time

        # Assert
        assert len(limiter._timestamps) == 5
        mock_sleep.assert_not_called()

    asyncio.run(run())

def test_acquire_waits_when_rate_limited(monkeypatch):
    async def run():
        tm = TimeMachine()
        monkeypatch.setattr(gptscan.time, 'monotonic', tm.time)

        limiter = gptscan.AsyncRateLimiter(rate_per_minute=1)

        # 1. First acquire (immediate)
        await limiter.acquire()
        assert len(limiter._timestamps) == 1

        # Advance time slightly (e.g., 10s elapsed since first request)
        tm.advance(10.0)

        # 2. Second acquire (should wait)
        # When sleep is called, we should advance time to simulate waiting
        async def side_effect_sleep(seconds):
            tm.advance(seconds)

        mock_sleep = MagicMock(side_effect=side_effect_sleep)
        monkeypatch.setattr(gptscan.asyncio, 'sleep', mock_sleep)

        await limiter.acquire()

        # Assert
        assert mock_sleep.call_count == 1
        args, _ = mock_sleep.call_args
        assert args[0] == pytest.approx(50.0)

        assert len(limiter._timestamps) == 1
        assert limiter._timestamps[0] == pytest.approx(160.0)

    asyncio.run(run())

def test_on_wait_callback(monkeypatch):
    async def run():
        tm = TimeMachine()
        monkeypatch.setattr(gptscan.time, 'monotonic', tm.time)

        # Mock sleep to just advance time without delay
        async def fast_sleep(seconds):
            tm.advance(seconds)

        monkeypatch.setattr(gptscan.asyncio, 'sleep', fast_sleep)

        limiter = gptscan.AsyncRateLimiter(rate_per_minute=1)
        callback = MagicMock()

        await limiter.acquire() # At 100.0
        tm.advance(1.0) # At 101.0

        await limiter.acquire(on_wait=callback)

        callback.assert_called_once()
        args, _ = callback.call_args
        assert args[0] == pytest.approx(59.0)

    asyncio.run(run())

def test_enforce_minimum_rate_limit():
    limiter = gptscan.AsyncRateLimiter(rate_per_minute=0)
    assert limiter.rate_per_minute == 1

    limiter = gptscan.AsyncRateLimiter(rate_per_minute=-5)
    assert limiter.rate_per_minute == 1

def test_sliding_window_cleanup(monkeypatch):
    async def run():
        tm = TimeMachine()
        monkeypatch.setattr(gptscan.time, 'monotonic', tm.time)
        monkeypatch.setattr(gptscan.asyncio, 'sleep', MagicMock())

        limiter = gptscan.AsyncRateLimiter(rate_per_minute=10)

        for _ in range(5):
            await limiter.acquire()
            tm.advance(1.0)

        assert len(limiter._timestamps) == 5

        tm.advance(60.0)

        await limiter.acquire()

        assert len(limiter._timestamps) == 1
        assert limiter._timestamps[0] == pytest.approx(165.0)

    asyncio.run(run())
