from pathlib import Path
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

NAV_PATH = PROJECT_DIR / "artifacts" / "unified_backtest_k4" / "plots" / "full_period_nav.csv"
OUTPUT_DIR = PROJECT_DIR / "artifacts" / "nav_animation"

TOP5_DIR = OUTPUT_DIR / "top5_frames"
ALL_DIR = OUTPUT_DIR / "all_frames"

TOP5_GIF = OUTPUT_DIR / "top5_dynamic_ranking.gif"
ALL_GIF = OUTPUT_DIR / "all_models_dynamic_ranking.gif"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TOP5_DIR.mkdir(parents=True, exist_ok=True)
ALL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load NAV data
# ============================================================

nav = pd.read_csv(NAV_PATH, index_col=0, parse_dates=True)

# remove columns that are all NaN
nav = nav.dropna(axis=1, how="all")

# forward fill missing values
nav = nav.ffill()

# remove rows where everything is still NaN
nav = nav.dropna(how="all")

# normalize each model by first valid value
normalized_nav = nav.copy()
for col in normalized_nav.columns:
    first_valid = normalized_nav[col].dropna()
    if len(first_valid) > 0:
        normalized_nav[col] = normalized_nav[col] / first_valid.iloc[0]

normalized_nav = normalized_nav.dropna(axis=1, how="all")


# ============================================================
# Helper: choose top models by final NAV
# ============================================================

def get_top_models(nav_df, n=5):
    final_values = nav_df.iloc[-1].dropna().sort_values(ascending=False)
    return list(final_values.head(n).index)


# ============================================================
# Helper: draw one frame
# ============================================================

def draw_frame(current_nav, all_model_names, frame_path, title, max_lines=10):
    current_date = current_nav.index[-1]

    ranking = current_nav.iloc[-1].dropna().sort_values(ascending=False)

    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3.4, 1.3])

    ax = fig.add_subplot(gs[0])
    ax_rank = fig.add_subplot(gs[1])

    # NAV lines
    for col in current_nav.columns:
        linewidth = 2.2 if col in ranking.head(5).index else 1.0
        alpha = 0.95 if col in ranking.head(5).index else 0.35

        if "baseline" in col.lower() or "equal" in col.lower() or "60" in col:
            ax.plot(
                current_nav.index,
                current_nav[col],
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                label=col
            )
        else:
            ax.plot(
                current_nav.index,
                current_nav[col],
                linewidth=linewidth,
                alpha=alpha,
                label=col
            )

    ax.set_title(f"{title}\nThrough {current_date.date()}", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized NAV")
    ax.grid(alpha=0.25)

    if len(current_nav.columns) <= 8:
        ax.legend(fontsize=7, loc="upper left")

    # Ranking panel
    ax_rank.axis("off")
    ax_rank.set_title("Current Ranking", fontsize=12, pad=15)

    ranking_text = ""
    show_ranking = ranking.head(max_lines)

    for i, (name, value) in enumerate(show_ranking.items(), start=1):
        ranking_text += f"{i}. {name}\n   NAV: {value:.3f}\n\n"

    ax_rank.text(
        0.02,
        0.98,
        ranking_text,
        transform=ax_rank.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        family="monospace"
    )

    plt.tight_layout()
    plt.savefig(frame_path, dpi=130, bbox_inches="tight")
    plt.close()


# ============================================================
# Helper: make gif
# ============================================================

def make_animation(nav_df, frame_dir, gif_path, title, step=10, start=50, max_rank_lines=10):
    frames = []

    for i in range(start, len(nav_df), step):
        current_nav = nav_df.iloc[:i].copy()

        if current_nav.dropna(how="all").empty:
            continue

        frame_path = frame_dir / f"frame_{i:04d}.png"

        draw_frame(
            current_nav=current_nav,
            all_model_names=nav_df.columns,
            frame_path=frame_path,
            title=title,
            max_lines=max_rank_lines
        )

        frames.append(Image.open(frame_path).convert("RGB"))

    if not frames:
        raise ValueError("No frames were generated. Check NAV data.")

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=130,
        loop=0
    )

    print(f"Saved GIF: {gif_path}")


# ============================================================
# Top 5 animation
# ============================================================

top5_models = get_top_models(normalized_nav, n=5)

# add useful baselines if they exist
baseline_cols = [
    c for c in normalized_nav.columns
    if "baseline" in c.lower() or "equal" in c.lower() or "60" in c
]

top5_cols = []
for c in top5_models + baseline_cols:
    if c in normalized_nav.columns and c not in top5_cols:
        top5_cols.append(c)

top5_nav = normalized_nav[top5_cols]

make_animation(
    nav_df=top5_nav,
    frame_dir=TOP5_DIR,
    gif_path=TOP5_GIF,
    title="Top 5 Strategies NAV Comparison",
    step=10,
    start=50,
    max_rank_lines=8
)


# ============================================================
# All models animation
# ============================================================

make_animation(
    nav_df=normalized_nav,
    frame_dir=ALL_DIR,
    gif_path=ALL_GIF,
    title="All Track A / Track B / Baseline Models NAV Comparison",
    step=10,
    start=50,
    max_rank_lines=12
)