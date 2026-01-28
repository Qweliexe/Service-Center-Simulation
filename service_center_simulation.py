import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

def exp_time(mean):
    return np.random.exponential(mean)

def mean_ci_95(x):
    x = np.array(x, dtype=float)
    n = len(x)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    m = float(np.mean(x))
    if n == 1:
        return (m, m, m)
    s = float(np.std(x, ddof=1))
    z = 1.96
    half = z * s / sqrt(n)
    return (m, m - half, m + half)

def handle_request(env, channel, agents, service_mean, max_wait, stats, abandon_on):
    arrival_time = env.now
    stats["arrivals"][channel] += 1

    req = agents.request()

    if abandon_on and (max_wait is not None):
        result = yield req | env.timeout(max_wait)
        if req not in result:
            stats["abandoned"][channel] += 1
            return
    else:
        yield req

    service_start = env.now
    stats["wait_times"][channel].append(service_start - arrival_time)

    service_time = exp_time(service_mean)
    stats["service_times"][channel].append(service_time)
    yield env.timeout(service_time)

    stats["busy_time"][channel] += service_time
    agents.release(req)
    stats["completed"][channel] += 1

def arrival_process(env, channel, agents, arrival_rate_per_hr, service_mean, max_wait, stats, abandon_on):
    mean_interarrival = 60.0 / arrival_rate_per_hr
    while True:
        yield env.timeout(exp_time(mean_interarrival))
        env.process(handle_request(env, channel, agents, service_mean, max_wait, stats, abandon_on))

def run_sim(sim_time_minutes, params, seed):
    np.random.seed(seed)
    env = simpy.Environment()

    phone_agents = simpy.Resource(env, capacity=params["agents"]["phone"])
    chat_agents = simpy.Resource(env, capacity=params["agents"]["chat"])
    email_agents = simpy.Resource(env, capacity=params["agents"]["email"])

    stats = {
        "arrivals": {"phone": 0, "chat": 0, "email": 0},
        "completed": {"phone": 0, "chat": 0, "email": 0},
        "abandoned": {"phone": 0, "chat": 0, "email": 0},
        "wait_times": {"phone": [], "chat": [], "email": []},
        "service_times": {"phone": [], "chat": [], "email": []},
        "busy_time": {"phone": 0.0, "chat": 0.0, "email": 0.0},
    }

    env.process(arrival_process(
        env, "phone", phone_agents,
        params["arrival_rates"]["phone"],
        params["service_means"]["phone"],
        params["max_wait"]["phone"],
        stats,
        params["abandon"]["phone"]
    ))

    env.process(arrival_process(
        env, "chat", chat_agents,
        params["arrival_rates"]["chat"],
        params["service_means"]["chat"],
        params["max_wait"]["chat"],
        stats,
        params["abandon"]["chat"]
    ))

    env.process(arrival_process(
        env, "email", email_agents,
        params["arrival_rates"]["email"],
        params["service_means"]["email"],
        params["max_wait"]["email"],
        stats,
        params["abandon"]["email"]
    ))

    env.run(until=sim_time_minutes)

    util = {}
    util["phone"] = stats["busy_time"]["phone"] / (params["agents"]["phone"] * sim_time_minutes) if params["agents"]["phone"] > 0 else np.nan
    util["chat"] = stats["busy_time"]["chat"] / (params["agents"]["chat"] * sim_time_minutes) if params["agents"]["chat"] > 0 else np.nan
    util["email"] = stats["busy_time"]["email"] / (params["agents"]["email"] * sim_time_minutes) if params["agents"]["email"] > 0 else np.nan

    results = {
        "avg_wait_phone": np.mean(stats["wait_times"]["phone"]) if stats["wait_times"]["phone"] else 0.0,
        "avg_wait_chat": np.mean(stats["wait_times"]["chat"]) if stats["wait_times"]["chat"] else 0.0,
        "avg_wait_email": np.mean(stats["wait_times"]["email"]) if stats["wait_times"]["email"] else 0.0,
        "abandon_phone_pct": (stats["abandoned"]["phone"] / stats["arrivals"]["phone"] * 100) if stats["arrivals"]["phone"] else 0.0,
        "abandon_chat_pct": (stats["abandoned"]["chat"] / stats["arrivals"]["chat"] * 100) if stats["arrivals"]["chat"] else 0.0,
        "abandon_email_pct": (stats["abandoned"]["email"] / stats["arrivals"]["email"] * 100) if stats["arrivals"]["email"] else 0.0,
        "util_phone": util["phone"] * 100,
        "util_chat": util["chat"] * 100,
        "util_email": util["email"] * 100,
        "util_overall": np.nanmean([util["phone"], util["chat"], util["email"]]) * 100,
    }
    return results

def scenario_params():
    baseline = {
        "arrival_rates": {"phone": 28, "chat": 22, "email": 18},
        "service_means": {"phone": 9, "chat": 6, "email": 11},
        "agents": {"phone": 4, "chat": 4, "email": 3},
        "max_wait": {"phone": 6, "chat": 4, "email": 35},
        "abandon": {"phone": True, "chat": True, "email": False},
    }

    peak = {
        "arrival_rates": {"phone": 42, "chat": 33, "email": 23},
        "service_means": {"phone": 9, "chat": 6, "email": 11},
        "agents": {"phone": 4, "chat": 4, "email": 3},
        "max_wait": {"phone": 6, "chat": 4, "email": 35},
        "abandon": {"phone": True, "chat": True, "email": False},
    }

    improvement = {
        "arrival_rates": {"phone": 42, "chat": 33, "email": 23},
        "service_means": {"phone": 9, "chat": 6, "email": 11},
        "agents": {"phone": 5, "chat": 5, "email": 3},
        "max_wait": {"phone": 6, "chat": 4, "email": 35},
        "abandon": {"phone": True, "chat": True, "email": False},
    }

    return {"Baseline": baseline, "Peak": peak, "Improvement": improvement}


def run_experiment(sim_time_minutes=480, replications=30, seed0=1000):
    scen = scenario_params()
    rows = []
    raw = {name: [] for name in scen.keys()}

    for name, params in scen.items():
        for r in range(replications):
            res = run_sim(sim_time_minutes, params, seed=seed0 + r)
            raw[name].append(res)

        df = pd.DataFrame(raw[name])

        out = {"Scenario": name, "Runs": replications, "SimTime(min)": sim_time_minutes}

        for col in ["avg_wait_phone", "avg_wait_chat", "avg_wait_email",
                    "abandon_phone_pct", "abandon_chat_pct",
                    "util_overall"]:
            m, lo, hi = mean_ci_95(df[col].tolist())
            out[col] = m
            out[col + "_CI_low"] = lo
            out[col + "_CI_high"] = hi

        rows.append(out)

    summary = pd.DataFrame(rows)
    return summary

def plot_charts(summary_df):
    scenarios = summary_df["Scenario"].tolist()

    phone_wait = summary_df["avg_wait_phone"].tolist()
    chat_wait = summary_df["avg_wait_chat"].tolist()
    email_wait = summary_df["avg_wait_email"].tolist()

    x = np.arange(len(scenarios))

    plt.figure()
    plt.plot(x, phone_wait, marker="o", label="Phone")
    plt.plot(x, chat_wait, marker="o", label="Live Chat")
    plt.plot(x, email_wait, marker="o", label="E-Mail")
    plt.xticks(x, scenarios)
    plt.ylabel("Average waiting time(min)")
    plt.title("Average Waiting Time According to Scenarios")
    plt.legend()
    plt.show()

    phone_ab = summary_df["abandon_phone_pct"].tolist()
    chat_ab = summary_df["abandon_chat_pct"].tolist()

    plt.figure()
    plt.plot(x, phone_ab, marker="o", label="abonden phone %")
    plt.plot(x, chat_ab, marker="o", label="abonden live chat %")
    plt.xticks(x, scenarios)
    plt.ylabel("abondonment rate(%)")
    plt.title("Abandonment Rate According to Scenarios")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    summary = run_experiment(sim_time_minutes=480, replications=30, seed0=1000)
    pd.set_option("display.max_columns", None)
    print("\n=== KPI Summary Table (Average and 95% CI) ===\n")
    print(summary.to_string(index=False))
    plot_charts(summary)
