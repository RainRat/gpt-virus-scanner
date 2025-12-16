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

@pytest.mark.asyncio
async def test_acquire_allows_immediate_access(mocker):
    # Setup
    tm = TimeMachine()
    mocker.patch('gptscan.time.monotonic', side_effect=tm.time)
    mock_sleep = mocker.patch('gptscan.asyncio.sleep')

    limiter = gptscan.AsyncRateLimiter(rate_per_minute=5)

    # Action: Acquire 5 times
    for _ in range(5):
        await limiter.acquire()
        tm.advance(0.1) # Simulate tiny processing time

    # Assert
    assert len(limiter._timestamps) == 5
    mock_sleep.assert_not_called()

@pytest.mark.asyncio
async def test_acquire_waits_when_rate_limited(mocker):
    # Setup
    tm = TimeMachine()
    mocker.patch('gptscan.time.monotonic', side_effect=tm.time)

    limiter = gptscan.AsyncRateLimiter(rate_per_minute=1)

    # 1. First acquire (immediate)
    await limiter.acquire()
    assert len(limiter._timestamps) == 1

    # Advance time slightly (e.g., 10s elapsed since first request)
    tm.advance(10.0)
    # Current time = 110. First request was at 100.
    # Window ends at 100 + 60 = 160.
    # Wait needed = 160 - 110 = 50.

    # 2. Second acquire (should wait)
    # When sleep is called, we should advance time to simulate waiting
    async def side_effect_sleep(seconds):
        tm.advance(seconds)

    mock_sleep = mocker.patch('gptscan.asyncio.sleep', side_effect=side_effect_sleep)

    await limiter.acquire()

    # Assert
    assert mock_sleep.call_count == 1
    # Check the wait time argument
    args, _ = mock_sleep.call_args
    assert args[0] == pytest.approx(50.0)

    # Verify timestamps
    # Should have 1 timestamp now (the new one).
    # The loop logic:
    # 1. Acquire lock.
    # 2. Check timestamps. Limit reached.
    # 3. Release lock, sleep(50).
    # 4. Loop back.
    # 5. Acquire lock.
    # 6. now is 160.
    # 7. old timestamp (100) is expired (160 - 100 >= 60). Pop.
    # 8. timestamps empty.
    # 9. Append 160.

    assert len(limiter._timestamps) == 1
    assert limiter._timestamps[0] == pytest.approx(160.0)


@pytest.mark.asyncio
async def test_on_wait_callback(mocker):
    tm = TimeMachine()
    mocker.patch('gptscan.time.monotonic', side_effect=tm.time)

    # Mock sleep to just advance time without delay
    async def fast_sleep(seconds):
        tm.advance(seconds)

    mocker.patch('gptscan.asyncio.sleep', side_effect=fast_sleep)

    limiter = gptscan.AsyncRateLimiter(rate_per_minute=1)
    callback = MagicMock()

    await limiter.acquire() # At 100.0
    tm.advance(1.0) # At 101.0

    await limiter.acquire(on_wait=callback)

    callback.assert_called_once()
    args, _ = callback.call_args
    # Wait time should be 60 - (101 - 100) = 59
    assert args[0] == pytest.approx(59.0)

@pytest.mark.asyncio
async def test_enforce_minimum_rate_limit():
    limiter = gptscan.AsyncRateLimiter(rate_per_minute=0)
    assert limiter.rate_per_minute == 1

    limiter = gptscan.AsyncRateLimiter(rate_per_minute=-5)
    assert limiter.rate_per_minute == 1

@pytest.mark.asyncio
async def test_sliding_window_cleanup(mocker):
    tm = TimeMachine()
    mocker.patch('gptscan.time.monotonic', side_effect=tm.time)
    mocker.patch('gptscan.asyncio.sleep')

    limiter = gptscan.AsyncRateLimiter(rate_per_minute=10)

    # Add 5 requests
    for _ in range(5):
        await limiter.acquire()
        tm.advance(1.0)
        # Requests at 100, 101, 102, 103, 104

    assert len(limiter._timestamps) == 5

    # Advance time significantly (past the window of the first requests)
    # Current time is 105.
    # Advance by 60 -> 165.
    tm.advance(60.0)

    # Next acquire
    await limiter.acquire()
    # At 165.
    # 100, 101, 102, 103, 104 are all > 60s old relative to 165
    # 165 - 100 = 65 >= 60. Pop.
    # ...
    # All should be popped.

    # Then 165 is appended.
    assert len(limiter._timestamps) == 1
    assert limiter._timestamps[0] == pytest.approx(165.0)
