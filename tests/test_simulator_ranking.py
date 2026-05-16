from agent.schemas import MarketStat, Prediction
from eval.simulator import simulate_return
from forecasting.calibration import market_probability


def test_simulator_ranks_strategies_sensibly() -> None:
    stats = {"Yes": MarketStat(last_price=0.5, yes_ask=0.51, no_ask=0.49)}
    outcome_yes = True
    p_market = market_probability(stats, "Yes")

    market_follower = Prediction(YES=p_market, NO=1.0 - p_market)
    correct_edge = Prediction(YES=0.7, NO=0.3)
    wrong_edge = Prediction(YES=0.3, NO=0.7)

    r_market = simulate_return(market_follower, stats, outcome_yes=outcome_yes)
    r_good = simulate_return(correct_edge, stats, outcome_yes=outcome_yes)
    r_bad = simulate_return(wrong_edge, stats, outcome_yes=outcome_yes)

    assert abs(r_market) < 1e-9
    assert r_good > r_market
    assert r_bad < r_market
