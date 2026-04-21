from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cluster_action_backtest import (
    ActionBacktestConfig,
    _objective_value,
    build_model_ranking_tables,
    calculate_backtest_metrics,
    save_evaluation_artifacts,
    split_window_dates,
    summarize_split_dates,
)


@dataclass
class QLearningAgentConfig:
    episodes: int = 250
    learning_rate: float = 0.10
    discount_factor: float = 0.95
    epsilon_start: float = 0.30
    epsilon_end: float = 0.02
    epsilon_decay: float = 0.985
    include_previous_action: bool = True
    eval_interval: int = 1
    random_state: int = 42
    reward_scale: float = 1.0


def _safe_experiment_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)


def prepare_rl_decision_data(
    cluster_series: pd.Series,
    asset_returns: pd.DataFrame,
    config: ActionBacktestConfig,
) -> tuple[pd.Series, pd.DataFrame]:
    common_dates = cluster_series.index.intersection(asset_returns.index)
    if common_dates.empty:
        raise ValueError("No overlapping dates between cluster_series and asset_returns for RL backtest.")

    aligned_clusters = cluster_series.loc[common_dates].astype(int)
    aligned_returns = asset_returns.loc[common_dates, list(config.tradable_assets)].copy()

    if config.execution_lag > 0:
        aligned_clusters = aligned_clusters.shift(config.execution_lag)

    decision_frame = pd.concat(
        [
            aligned_clusters.rename("cluster"),
            aligned_returns,
        ],
        axis=1,
    ).dropna(how="any")

    if decision_frame.empty:
        raise ValueError(
            "No valid RL decision rows remain after applying execution lag and aligning returns."
        )

    decision_clusters = decision_frame["cluster"].astype(int)
    decision_returns = decision_frame.loc[:, list(config.tradable_assets)].copy()
    decision_clusters.index.name = "Date"
    decision_returns.index.name = "Date"
    return decision_clusters, decision_returns


class DiscreteActionBacktestEnv:
    def __init__(
        self,
        cluster_series: pd.Series,
        asset_returns: pd.DataFrame,
        action_library: dict[str, pd.Series],
        config: ActionBacktestConfig,
        include_previous_action: bool = True,
        cluster_to_idx: dict[int, int] | None = None,
    ) -> None:
        if cluster_series.empty:
            raise ValueError("cluster_series is empty.")
        if asset_returns.empty:
            raise ValueError("asset_returns is empty.")

        common_dates = cluster_series.index.intersection(asset_returns.index)
        if common_dates.empty:
            raise ValueError("No overlapping dates between cluster_series and asset_returns.")

        self.cluster_series = cluster_series.loc[common_dates].astype(int).copy()
        self.asset_returns = asset_returns.loc[common_dates, list(config.tradable_assets)].copy()
        self.action_names = list(action_library.keys())
        self.action_weights = np.vstack(
            [
                action_library[action_name].reindex(config.tradable_assets).fillna(0.0).to_numpy(dtype=float)
                for action_name in self.action_names
            ]
        )
        self.config = config
        self.include_previous_action = include_previous_action

        if cluster_to_idx is None:
            cluster_ids = sorted(int(cluster_id) for cluster_id in self.cluster_series.unique())
            self.cluster_to_idx = {cluster_id: idx for idx, cluster_id in enumerate(cluster_ids)}
        else:
            self.cluster_to_idx = {int(cluster_id): int(idx) for cluster_id, idx in cluster_to_idx.items()}
            missing_cluster_ids = sorted(
                int(cluster_id)
                for cluster_id in self.cluster_series.unique()
                if int(cluster_id) not in self.cluster_to_idx
            )
            if missing_cluster_ids:
                raise ValueError(
                    "cluster_to_idx is missing cluster ids present in the evaluation data: "
                    f"{missing_cluster_ids}"
                )
        self.idx_to_cluster = {idx: cluster_id for cluster_id, idx in self.cluster_to_idx.items()}

        self.n_actions = len(self.action_names)
        self.previous_action_cardinality = self.n_actions + 1 if include_previous_action else 1
        self.n_states = len(self.cluster_to_idx) * self.previous_action_cardinality

        self._dates = pd.Index(self.cluster_series.index, name="Date")
        self._step_idx = 0
        self._previous_action_code = 0
        self._previous_weights = np.zeros(len(config.tradable_assets), dtype=float)

    def _encode_state(self, cluster_value: int, previous_action_code: int) -> int:
        cluster_idx = self.cluster_to_idx[int(cluster_value)]
        return cluster_idx * self.previous_action_cardinality + previous_action_code

    def reset(self) -> int:
        self._step_idx = 0
        self._previous_action_code = 0
        self._previous_weights = np.zeros(len(self.config.tradable_assets), dtype=float)
        first_cluster = int(self.cluster_series.iloc[0])
        return self._encode_state(first_cluster, self._previous_action_code)

    def step(self, action_idx: int) -> tuple[int | None, float, bool, dict]:
        if action_idx < 0 or action_idx >= self.n_actions:
            raise ValueError(f"action_idx={action_idx} is out of bounds for {self.n_actions} actions.")

        current_date = self._dates[self._step_idx]
        current_cluster = int(self.cluster_series.iloc[self._step_idx])
        current_returns = self.asset_returns.iloc[self._step_idx].to_numpy(dtype=float)
        current_weights = self.action_weights[action_idx]

        turnover = float(np.abs(current_weights - self._previous_weights).sum())
        gross_return = float(np.dot(current_weights, current_returns))
        transaction_cost = float(turnover * self.config.transaction_cost)
        reward = float(gross_return - transaction_cost)

        self._previous_weights = current_weights.copy()
        self._previous_action_code = action_idx + 1 if self.include_previous_action else 0
        self._step_idx += 1
        done = self._step_idx >= len(self._dates)

        next_state = None
        if not done:
            next_cluster = int(self.cluster_series.iloc[self._step_idx])
            next_state = self._encode_state(next_cluster, self._previous_action_code)

        info = {
            "date": current_date,
            "cluster": current_cluster,
            "action_idx": int(action_idx),
            "action_name": self.action_names[action_idx],
            "gross_return": gross_return,
            "transaction_cost": transaction_cost,
            "portfolio_return": gross_return - transaction_cost,
            "turnover": turnover,
            "weights": current_weights.copy(),
        }
        return next_state, reward, done, info


class TabularQLearningAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float,
        discount_factor: float,
        random_state: int,
    ) -> None:
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.learning_rate = float(learning_rate)
        self.discount_factor = float(discount_factor)
        self.rng = np.random.default_rng(random_state)
        self.q_table = np.zeros((self.n_states, self.n_actions), dtype=float)

    def select_action(self, state: int, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))

        state_values = self.q_table[state]
        max_value = state_values.max()
        best_actions = np.flatnonzero(np.isclose(state_values, max_value))
        return int(self.rng.choice(best_actions))

    def greedy_action(self, state: int) -> int:
        state_values = self.q_table[state]
        return int(np.argmax(state_values))

    def update(self, state: int, action: int, reward: float, next_state: int | None, done: bool) -> None:
        current_value = self.q_table[state, action]
        next_value = 0.0 if done or next_state is None else float(self.q_table[next_state].max())
        target = reward + self.discount_factor * next_value
        self.q_table[state, action] = current_value + self.learning_rate * (target - current_value)


def run_q_policy_backtest(
    cluster_series: pd.Series,
    asset_returns: pd.DataFrame,
    action_library: dict[str, pd.Series],
    config: ActionBacktestConfig,
    q_table: np.ndarray,
    include_previous_action: bool = True,
    cluster_to_idx: dict[int, int] | None = None,
) -> dict:
    env = DiscreteActionBacktestEnv(
        cluster_series=cluster_series,
        asset_returns=asset_returns,
        action_library=action_library,
        config=config,
        include_previous_action=include_previous_action,
        cluster_to_idx=cluster_to_idx,
    )

    records: list[dict] = []
    state = env.reset()
    done = False
    while not done:
        action_idx = int(np.argmax(q_table[state]))
        next_state, _, done, info = env.step(action_idx)
        records.append(info)
        if next_state is None:
            break
        state = next_state

    records_df = pd.DataFrame(records)
    if records_df.empty:
        raise ValueError("No RL backtest records were generated.")

    records_df["Date"] = pd.to_datetime(records_df["date"])
    records_df = records_df.set_index("Date").sort_index()

    assets = list(config.tradable_assets)
    target_weights = pd.DataFrame(
        np.vstack(records_df["weights"].to_numpy()),
        index=records_df.index,
        columns=assets,
    )
    portfolio_returns = records_df["portfolio_return"].astype(float).rename("portfolio_return")
    portfolio_nav = ((1.0 + portfolio_returns).cumprod() * config.initial_capital).rename("portfolio_nav")
    gross_returns = records_df["gross_return"].astype(float).rename("gross_return")
    transaction_costs = records_df["transaction_cost"].astype(float).rename("transaction_cost")
    turnover = records_df["turnover"].astype(float).rename("turnover")
    cluster_output = records_df["cluster"].astype(int).rename("cluster")
    action_output = records_df["action_name"].astype(str).rename("action")

    return {
        "cluster_series": cluster_output,
        "action_series": action_output,
        "target_weights": target_weights,
        "executed_weights": target_weights.copy(),
        "asset_returns": asset_returns.loc[target_weights.index, assets].copy(),
        "gross_returns": gross_returns,
        "transaction_costs": transaction_costs,
        "portfolio_returns": portfolio_returns,
        "portfolio_nav": portfolio_nav,
        "turnover": turnover,
    }


def train_q_learning_agent(
    train_clusters: pd.Series,
    train_returns: pd.DataFrame,
    validation_clusters: pd.Series,
    validation_returns: pd.DataFrame,
    action_library: dict[str, pd.Series],
    backtest_config: ActionBacktestConfig,
    agent_config: QLearningAgentConfig,
    cluster_to_idx: dict[int, int] | None = None,
) -> dict:
    train_env = DiscreteActionBacktestEnv(
        cluster_series=train_clusters,
        asset_returns=train_returns,
        action_library=action_library,
        config=backtest_config,
        include_previous_action=agent_config.include_previous_action,
        cluster_to_idx=cluster_to_idx,
    )
    agent = TabularQLearningAgent(
        n_states=train_env.n_states,
        n_actions=train_env.n_actions,
        learning_rate=agent_config.learning_rate,
        discount_factor=agent_config.discount_factor,
        random_state=agent_config.random_state,
    )

    history_rows: list[dict] = []
    epsilon = float(agent_config.epsilon_start)
    best_q_table = agent.q_table.copy()
    best_validation_metrics: dict | None = None
    best_objective_value = -np.inf
    best_episode = 0

    for episode in range(1, agent_config.episodes + 1):
        state = train_env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0

        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = train_env.step(action)
            scaled_reward = float(reward * agent_config.reward_scale)
            agent.update(state, action, scaled_reward, next_state, done)
            episode_reward += scaled_reward
            step_count += 1
            if next_state is None:
                break
            state = next_state

        row = {
            "episode": episode,
            "epsilon": epsilon,
            "train_episode_reward": episode_reward,
            "train_avg_reward": episode_reward / max(step_count, 1),
        }

        if episode % max(agent_config.eval_interval, 1) == 0:
            validation_backtest = run_q_policy_backtest(
                cluster_series=validation_clusters,
                asset_returns=validation_returns,
                action_library=action_library,
                config=backtest_config,
                q_table=agent.q_table,
                include_previous_action=agent_config.include_previous_action,
                cluster_to_idx=train_env.cluster_to_idx,
            )
            validation_metrics = calculate_backtest_metrics(
                portfolio_returns=validation_backtest["portfolio_returns"],
                turnover=validation_backtest["turnover"],
                config=backtest_config,
            )
            validation_score = _objective_value(validation_metrics, backtest_config.objective)
            row.update({f"validation_{key}": value for key, value in validation_metrics.items()})
            row["validation_objective_value"] = validation_score

            if validation_score > best_objective_value:
                best_objective_value = validation_score
                best_q_table = agent.q_table.copy()
                best_validation_metrics = validation_metrics
                best_episode = episode

        history_rows.append(row)
        epsilon = max(agent_config.epsilon_end, epsilon * agent_config.epsilon_decay)

    history_df = pd.DataFrame(history_rows)
    if best_validation_metrics is None:
        validation_backtest = run_q_policy_backtest(
            cluster_series=validation_clusters,
            asset_returns=validation_returns,
            action_library=action_library,
            config=backtest_config,
            q_table=best_q_table,
            include_previous_action=agent_config.include_previous_action,
            cluster_to_idx=train_env.cluster_to_idx,
        )
        best_validation_metrics = calculate_backtest_metrics(
            portfolio_returns=validation_backtest["portfolio_returns"],
            turnover=validation_backtest["turnover"],
            config=backtest_config,
        )
        best_objective_value = _objective_value(best_validation_metrics, backtest_config.objective)

    return {
        "best_q_table": best_q_table,
        "history_df": history_df,
        "best_episode": int(best_episode),
        "best_validation_metrics": best_validation_metrics,
        "best_validation_objective": float(best_objective_value),
        "n_states": int(train_env.n_states),
        "n_actions": int(train_env.n_actions),
        "cluster_to_idx": train_env.cluster_to_idx,
        "action_names": train_env.action_names,
    }


def evaluate_model_with_q_learning(
    experiment_result: dict,
    asset_returns: pd.DataFrame,
    action_library: dict[str, pd.Series],
    backtest_config: ActionBacktestConfig,
    agent_config: QLearningAgentConfig,
    pretrained_policy: dict | None = None,
) -> dict:
    window_end_dates = pd.Index(experiment_result["window_end_dates"], name="Date")
    raw_cluster_series = pd.Series(
        experiment_result["cluster_labels"],
        index=window_end_dates,
        name="cluster",
        dtype=int,
    )
    if "splits" in experiment_result and experiment_result["splits"] is not None:
        splits = {
            split_name: pd.Index(split_dates, name="Date")
            for split_name, split_dates in experiment_result["splits"].items()
        }
        split_summary = experiment_result.get("split_summary", summarize_split_dates(splits))
    else:
        splits = split_window_dates(
            window_end_dates=window_end_dates,
            validation_ratio=backtest_config.validation_ratio,
            test_ratio=backtest_config.test_ratio,
            random_state=backtest_config.split_random_state,
        )
        split_summary = summarize_split_dates(splits)

    decision_clusters, decision_returns = prepare_rl_decision_data(
        cluster_series=raw_cluster_series,
        asset_returns=asset_returns,
        config=backtest_config,
    )

    train_dates = decision_clusters.index.intersection(splits["train"])
    validation_dates = decision_clusters.index.intersection(splits["validation"])
    test_dates = decision_clusters.index.intersection(splits["test"])
    if train_dates.empty or validation_dates.empty or test_dates.empty:
        raise ValueError(
            "Train/validation/test decision dates are empty after RL alignment. "
            "Try reducing execution_lag or checking date overlap."
        )

    global_cluster_ids = sorted(int(cluster_id) for cluster_id in decision_clusters.unique())
    cluster_to_idx = {cluster_id: idx for idx, cluster_id in enumerate(global_cluster_ids)}

    if pretrained_policy is None:
        training_result = train_q_learning_agent(
            train_clusters=decision_clusters.loc[train_dates],
            train_returns=decision_returns.loc[train_dates],
            validation_clusters=decision_clusters.loc[validation_dates],
            validation_returns=decision_returns.loc[validation_dates],
            action_library=action_library,
            backtest_config=backtest_config,
            agent_config=agent_config,
            cluster_to_idx=cluster_to_idx,
        )
        best_q_table = training_result["best_q_table"]
        policy_loaded = False
    else:
        best_q_table = np.asarray(pretrained_policy["q_table"], dtype=float)
        training_result = {
            "best_q_table": best_q_table,
            "history_df": pd.DataFrame(),
            "best_episode": int(pretrained_policy.get("best_episode", 0)),
            "best_validation_metrics": None,
            "best_validation_objective": np.nan,
            "n_states": int(best_q_table.shape[0]),
            "n_actions": int(best_q_table.shape[1]),
            "cluster_to_idx": {int(k): int(v) for k, v in pretrained_policy["cluster_to_idx"].items()},
            "action_names": list(pretrained_policy["action_names"]),
        }
        cluster_to_idx = training_result["cluster_to_idx"]
        policy_loaded = True

    full_backtest = run_q_policy_backtest(
        cluster_series=decision_clusters,
        asset_returns=decision_returns,
        action_library=action_library,
        config=backtest_config,
        q_table=best_q_table,
        include_previous_action=agent_config.include_previous_action,
        cluster_to_idx=cluster_to_idx,
    )
    train_backtest = run_q_policy_backtest(
        cluster_series=decision_clusters.loc[train_dates],
        asset_returns=decision_returns.loc[train_dates],
        action_library=action_library,
        config=backtest_config,
        q_table=best_q_table,
        include_previous_action=agent_config.include_previous_action,
        cluster_to_idx=cluster_to_idx,
    )
    validation_backtest = run_q_policy_backtest(
        cluster_series=decision_clusters.loc[validation_dates],
        asset_returns=decision_returns.loc[validation_dates],
        action_library=action_library,
        config=backtest_config,
        q_table=best_q_table,
        include_previous_action=agent_config.include_previous_action,
        cluster_to_idx=cluster_to_idx,
    )
    test_backtest = run_q_policy_backtest(
        cluster_series=decision_clusters.loc[test_dates],
        asset_returns=decision_returns.loc[test_dates],
        action_library=action_library,
        config=backtest_config,
        q_table=best_q_table,
        include_previous_action=agent_config.include_previous_action,
        cluster_to_idx=cluster_to_idx,
    )

    train_metrics = calculate_backtest_metrics(
        portfolio_returns=train_backtest["portfolio_returns"],
        turnover=train_backtest["turnover"],
        config=backtest_config,
    )
    validation_metrics = calculate_backtest_metrics(
        portfolio_returns=validation_backtest["portfolio_returns"],
        turnover=validation_backtest["turnover"],
        config=backtest_config,
    )
    test_metrics = calculate_backtest_metrics(
        portfolio_returns=test_backtest["portfolio_returns"],
        turnover=test_backtest["turnover"],
        config=backtest_config,
    )

    policy_name = (
        "q_learning_policy("
        f"episodes={agent_config.episodes},"
        f"alpha={agent_config.learning_rate:.3f},"
        f"gamma={agent_config.discount_factor:.3f}"
        ")"
    )
    experiment_name = f"{experiment_result['experiment_name']}__q_learning"
    summary = {
        "experiment_name": experiment_name,
        "source_experiment": experiment_result["experiment_name"],
        "architecture": f"{experiment_result['summary'].get('architecture')}+q_learning",
        "n_clusters": int(np.unique(decision_clusters.values).size),
        "target_cluster_count": int(experiment_result["summary"].get("target_cluster_count", np.unique(decision_clusters.values).size)),
        "best_mapping": policy_name,
        "objective": backtest_config.objective,
        "best_episode": int(training_result["best_episode"]),
        "n_states": int(training_result["n_states"]),
        "n_actions": int(training_result["n_actions"]),
        "policy_loaded": bool(policy_loaded),
    }
    summary.update({f"train_{key}": value for key, value in train_metrics.items()})
    summary.update({f"validation_{key}": value for key, value in validation_metrics.items()})
    summary.update({f"test_{key}": value for key, value in test_metrics.items()})

    q_table_df = pd.DataFrame(best_q_table, columns=training_result["action_names"])
    q_table_df.index.name = "state_id"

    return {
        "summary": summary,
        "splits": splits,
        "split_summary": split_summary,
        "best_mapping": None,
        "best_mapping_text": policy_name,
        "search_results": training_result["history_df"],
        "backtest_result": full_backtest,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "q_table": q_table_df,
        "cluster_to_idx": training_result["cluster_to_idx"],
        "action_names": training_result["action_names"],
        "agent_config": agent_config,
        "source_experiment": experiment_result["experiment_name"],
    }


def save_q_learning_policy(
    output_dir: str | Path,
    evaluation_result: dict,
) -> dict | None:
    q_table = evaluation_result.get("q_table")
    cluster_to_idx = evaluation_result.get("cluster_to_idx")
    action_names = evaluation_result.get("action_names")
    agent_config = evaluation_result.get("agent_config")
    if q_table is None or cluster_to_idx is None or action_names is None or agent_config is None:
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    experiment_name = evaluation_result["summary"]["experiment_name"]
    safe_name = _safe_experiment_name(experiment_name)
    q_table_path = output_path / f"{safe_name}_q_table.csv"
    metadata_path = output_path / f"{safe_name}_metadata.json"

    if isinstance(q_table, pd.DataFrame):
        q_table.to_csv(q_table_path)
        action_names = list(q_table.columns)
    else:
        q_table_frame = pd.DataFrame(np.asarray(q_table, dtype=float), columns=list(action_names))
        q_table_frame.index.name = "state_id"
        q_table_frame.to_csv(q_table_path)

    metadata = {
        "experiment_name": experiment_name,
        "source_experiment": evaluation_result["summary"].get("source_experiment"),
        "architecture": evaluation_result["summary"].get("architecture"),
        "best_episode": int(evaluation_result["summary"].get("best_episode", 0)),
        "action_names": list(action_names),
        "cluster_to_idx": {str(k): int(v) for k, v in cluster_to_idx.items()},
        "agent_config": {
            "episodes": int(agent_config.episodes),
            "learning_rate": float(agent_config.learning_rate),
            "discount_factor": float(agent_config.discount_factor),
            "epsilon_start": float(agent_config.epsilon_start),
            "epsilon_end": float(agent_config.epsilon_end),
            "epsilon_decay": float(agent_config.epsilon_decay),
            "include_previous_action": bool(agent_config.include_previous_action),
            "eval_interval": int(agent_config.eval_interval),
            "random_state": int(agent_config.random_state),
            "reward_scale": float(agent_config.reward_scale),
        },
        "summary": evaluation_result["summary"],
        "q_table_filename": q_table_path.name,
        "q_table_path": str(q_table_path.resolve()),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "experiment_name": experiment_name,
        "q_table_path": str(q_table_path),
        "metadata_path": str(metadata_path),
    }


def load_q_learning_policy(
    metadata_path: str | Path,
) -> dict:
    metadata_file = Path(metadata_path)
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    q_table_filename = metadata.get("q_table_filename")
    if q_table_filename is not None:
        q_table_path = metadata_file.parent / q_table_filename
    else:
        q_table_path = Path(metadata["q_table_path"])
        if not q_table_path.is_absolute():
            q_table_path = metadata_file.parent / q_table_path
    q_table_frame = pd.read_csv(q_table_path, index_col=0)
    return {
        "experiment_name": metadata["experiment_name"],
        "source_experiment": metadata.get("source_experiment"),
        "architecture": metadata.get("architecture"),
        "best_episode": int(metadata.get("best_episode", 0)),
        "action_names": list(metadata["action_names"]),
        "cluster_to_idx": {int(k): int(v) for k, v in metadata["cluster_to_idx"].items()},
        "agent_config": metadata.get("agent_config", {}),
        "summary": metadata.get("summary", {}),
        "q_table": q_table_frame.to_numpy(dtype=float),
        "q_table_frame": q_table_frame,
        "metadata_path": str(metadata_file),
        "q_table_path": str(q_table_path),
    }


def save_rl_ranking_artifacts(
    output_dir: str | Path,
    evaluation_results: list[dict],
    validation_ranking: pd.DataFrame,
    test_ranking: pd.DataFrame,
    combined_ranking: pd.DataFrame,
) -> None:
    save_evaluation_artifacts(
        output_dir=output_dir,
        evaluation_results=evaluation_results,
        validation_ranking=validation_ranking,
        test_ranking=test_ranking,
        combined_ranking=combined_ranking,
    )

    output_path = Path(output_dir)
    q_table_dir = output_path / "q_tables"
    q_table_dir.mkdir(parents=True, exist_ok=True)
    metadata_rows = []

    for result in evaluation_results:
        policy_manifest_row = save_q_learning_policy(q_table_dir, result)
        if policy_manifest_row is not None:
            metadata_rows.append(policy_manifest_row)

    if metadata_rows:
        pd.DataFrame(metadata_rows).to_csv(q_table_dir / "q_table_manifest.csv", index=False)


__all__ = [
    "QLearningAgentConfig",
    "prepare_rl_decision_data",
    "DiscreteActionBacktestEnv",
    "TabularQLearningAgent",
    "run_q_policy_backtest",
    "train_q_learning_agent",
    "evaluate_model_with_q_learning",
    "save_q_learning_policy",
    "load_q_learning_policy",
    "save_rl_ranking_artifacts",
    "build_model_ranking_tables",
]
