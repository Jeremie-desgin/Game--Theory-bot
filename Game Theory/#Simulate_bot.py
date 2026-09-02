# Simulate_Bot.py



def drifting_opponent(rnd, bot_h, opp_h, rounds=40000):
   
    defect_prob = rnd / rounds
    return 1 if np.random.random() < defect_prob else 0

OPPONENTS = {
    "Always Cooperate": always_cooperate,
    "Always Defect":    always_defect,
    "Random":           random_opponent,
    "Tit-for-Tat":      tit_for_tat,
    "Grudger":          grudger,
    "Switching":        switching_opponent,
    "Drifting":         drifting_opponent,
}


ROUNDS        = 40000
WINDOW        = 500     
SEED          = 42

def run_simulation(opponent_fn, payoff_matrix, max_reward, rounds=ROUNDS):
    
    bot = AdaptivePrisonerBot(seed=SEED)
    bot_score = 0
    opp_score = 0
    coop_rates = []

    for rnd in range(1, rounds + 1):
        opp_move = opponent_fn(rnd, bot.bot_history, bot.opp_history)
        bot_move = bot.act()

        bot_payoff, opp_payoff = payoff_matrix[(bot_move, opp_move)]
        bot.learn(bot_move, opp_move, bot_payoff, max_reward=max_reward)

        bot_score += bot_payoff
        opp_score += opp_payoff

        if rnd % WINDOW == 0:
            coop_rates.append(bot.coop_rate_window(WINDOW))

    final_coop = bot.coop_rate_window(WINDOW)
    return bot_score, opp_score, coop_rates, final_coop



print("=" * 60)
print("EXPERIMENT 1: All opponents — Standard PD payoff matrix")
print("=" * 60)

matrix_name = "Standard PD"
payoff_matrix = PAYOFF_MATRICES[matrix_name]
max_reward = MAX_REWARD[matrix_name]

exp1_results = {}
for opp_name, opp_fn in OPPONENTS.items():
    bot_score, opp_score, coop_rates, final_coop = run_simulation(
        opp_fn, payoff_matrix, max_reward
    )
    exp1_results[opp_name] = {
        "bot_score": bot_score,
        "opp_score": opp_score,
        "coop_rates": coop_rates,
        "final_coop": final_coop,
    }
    print(f"  vs {opp_name:20s} | Bot: {bot_score:7d} | Opp: {opp_score:7d} | "
          f"Final coop: {final_coop*100:5.1f}%")

fig, ax = plt.subplots(figsize=(12, 6))
x_axis = [(i + 1) * WINDOW for i in range(len(next(iter(exp1_results.values()))["coop_rates"]))]

colors = plt.cm.tab10.colors
for i, (opp_name, data) in enumerate(exp1_results.items()):
    style = "--" if opp_name in ("Switching", "Drifting") else "-"
    ax.plot(x_axis, data["coop_rates"], label=opp_name,
            color=colors[i % 10], linestyle=style, linewidth=1.8)

ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.6, label="50% baseline")
ax.axvline(20000, color="red", linestyle=":", linewidth=1.2, alpha=0.5, label="Switch point (Switching)")
ax.set_xlabel(f"Round (window = {WINDOW})", fontsize=12)
ax.set_ylabel("Cooperation Rate", fontsize=12)
ax.set_title("Experiment 1: Bot Cooperation Rate vs Each Opponent\n(Standard PD, 40 000 rounds)", fontsize=13)
ax.legend(loc="lower right", fontsize=9)
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(0, ROUNDS)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("results/exp1_cooperation_rates.png", dpi=150)
plt.close(fig)
print("  → Saved: results/exp1_cooperation_rates.png\n")



print("=" * 60)
print("EXPERIMENT 2: Tit-for-Tat vs all payoff matrices")
print("=" * 60)

exp2_results = {}
for mat_name, payoff_matrix in PAYOFF_MATRICES.items():
    max_reward = MAX_REWARD[mat_name]
    bot_score, opp_score, coop_rates, final_coop = run_simulation(
        tit_for_tat, payoff_matrix, max_reward
    )
    exp2_results[mat_name] = {
        "bot_score": bot_score,
        "opp_score": opp_score,
        "coop_rates": coop_rates,
        "final_coop": final_coop,
    }
    print(f"  {mat_name:15s} | Bot: {bot_score:7d} | Opp: {opp_score:7d} | "
          f"Final coop: {final_coop*100:5.1f}%")

fig, ax = plt.subplots(figsize=(12, 6))
for i, (mat_name, data) in enumerate(exp2_results.items()):
    ax.plot(x_axis, data["coop_rates"], label=mat_name,
            color=colors[i % 10], linewidth=2)

ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.6)
ax.set_xlabel(f"Round (window = {WINDOW})", fontsize=12)
ax.set_ylabel("Cooperation Rate", fontsize=12)
ax.set_title("Experiment 2: Effect of Payoff Matrix on Bot Cooperation\n(Opponent: Tit-for-Tat, 40 000 rounds)", fontsize=13)
ax.legend(fontsize=10)
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(0, ROUNDS)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("results/exp2_payoff_matrix_effect.png", dpi=150)
plt.close(fig)
print("  → Saved: results/exp2_payoff_matrix_effect.png\n")



print("=" * 60)
print("EXPERIMENT 3: Learning rate sensitivity — Switching opponent")
print("=" * 60)

payoff_matrix = PAYOFF_MATRICES["Standard PD"]
max_reward    = MAX_REWARD["Standard PD"]
learning_rates = [0.05, 0.1, 0.3, 0.5, 0.7]

exp3_results = {}
for lr in learning_rates:
    bot = AdaptivePrisonerBot(lr=lr, seed=SEED)
    bot_score = 0
    coop_rates = []

    for rnd in range(1, ROUNDS + 1):
        opp_move = switching_opponent(rnd, bot.bot_history, bot.opp_history)
        bot_move = bot.act()
        bot_payoff, _ = payoff_matrix[(bot_move, opp_move)]
        bot.learn(bot_move, opp_move, bot_payoff, max_reward=max_reward)
        bot_score += bot_payoff

        if rnd % WINDOW == 0:
            coop_rates.append(bot.coop_rate_window(WINDOW))

    exp3_results[lr] = {"bot_score": bot_score, "coop_rates": coop_rates}
    final = coop_rates[-1] if coop_rates else 0
    print(f"  lr={lr:.2f} | Bot score: {bot_score:7d} | Final coop: {final*100:5.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for i, (lr, data) in enumerate(exp3_results.items()):
    axes[0].plot(x_axis, data["coop_rates"], label=f"lr={lr}",
                 color=colors[i % 10], linewidth=2)

axes[0].axvline(20000, color="red", linestyle="--", linewidth=1.5,
                alpha=0.7, label="Strategy switch")
axes[0].axhline(0.5, color="gray", linestyle=":", alpha=0.5)
axes[0].set_xlabel(f"Round (window = {WINDOW})", fontsize=11)
axes[0].set_ylabel("Cooperation Rate", fontsize=11)
axes[0].set_title("Cooperation Rate over Time\n(Switching Opponent)", fontsize=12)
axes[0].legend(fontsize=9)
axes[0].set_ylim(-0.05, 1.05)
axes[0].grid(True, alpha=0.3)

scores = [d["bot_score"] for d in exp3_results.values()]
bar_colors = [colors[i % 10] for i in range(len(learning_rates))]
axes[1].bar([str(lr) for lr in learning_rates], scores,
            color=bar_colors, edgecolor="white", linewidth=0.5)
axes[1].set_xlabel("Learning Rate", fontsize=11)
axes[1].set_ylabel("Total Bot Score", fontsize=11)
axes[1].set_title("Total Score by Learning Rate\n(Switching Opponent)", fontsize=12)
axes[1].grid(True, alpha=0.3, axis="y")

fig.suptitle("Experiment 3: Learning Rate vs Adaptation Speed to Strategy Shift",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("results/exp3_learning_rate_sensitivity.png", dpi=150)
plt.close(fig)
print("  → Saved: results/exp3_learning_rate_sensitivity.png\n")



print("=" * 60)
print("EXPERIMENT 4: Tracking a gradually drifting opponent")
print("=" * 60)

payoff_matrix = PAYOFF_MATRICES["Standard PD"]
max_reward    = MAX_REWARD["Standard PD"]

bot = AdaptivePrisonerBot(seed=SEED)
bot_coop_rates = []
opp_coop_rates = []
bot_score = 0
opp_score = 0

bot_window_moves = []
opp_window_moves = []

for rnd in range(1, ROUNDS + 1):
    opp_move = drifting_opponent(rnd, bot.bot_history, bot.opp_history)
    bot_move = bot.act()

    bot_payoff, opp_payoff = payoff_matrix[(bot_move, opp_move)]
    bot.learn(bot_move, opp_move, bot_payoff, max_reward=max_reward)

    bot_score += bot_payoff
    opp_score += opp_payoff
    bot_window_moves.append(bot_move)
    opp_window_moves.append(opp_move)

    if rnd % WINDOW == 0:
        bot_coop_rates.append(bot_window_moves.count(0) / WINDOW)
        opp_coop_rates.append(opp_window_moves.count(0) / WINDOW)
        bot_window_moves = []
        opp_window_moves = []

print(f"  Bot score: {bot_score} | Opp score: {opp_score}")
print(f"  Final bot coop rate: {bot_coop_rates[-1]*100:.1f}%")
print(f"  Final opp coop rate: {opp_coop_rates[-1]*100:.1f}%")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x_axis, opp_coop_rates, label="Drifting Opponent (true coop rate)",
        color="tomato", linestyle="--", linewidth=2)
ax.plot(x_axis, bot_coop_rates, label="Bot cooperation rate",
        color="steelblue", linewidth=2)
ax.fill_between(x_axis, bot_coop_rates, opp_coop_rates, alpha=0.12, color="gray")
ax.set_xlabel(f"Round (window = {WINDOW})", fontsize=12)
ax.set_ylabel("Cooperation Rate", fontsize=12)
ax.set_title("Experiment 4: Bot Tracking a Gradually Drifting Opponent\n"
             "(Opponent defection probability increases linearly 0→1)", fontsize=13)
ax.legend(fontsize=11)
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(0, ROUNDS)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("results/exp4_drifting_opponent.png", dpi=150)
plt.close(fig)
print("  → Saved: results/exp4_drifting_opponent.png\n")



print("=" * 60)
print("SUMMARY — Experiment 1 final cooperation rates (Standard PD)")
print("=" * 60)
for opp_name, data in exp1_results.items():
    bar = "█" * int(data["final_coop"] * 20)
    print(f"  {opp_name:20s} {bar:20s} {data['final_coop']*100:5.1f}%")

print("\nAll experiments complete. Charts saved to ./results/")