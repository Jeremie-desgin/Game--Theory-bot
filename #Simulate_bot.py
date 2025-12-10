#Simulate_bot
import numpy as np
import random 
from LrBot import AdaptivePrisonerBot

def random_opponent(history):
    return np.random.choice([0,1])

def tit_for_tat(opp_history):
    if not opp_history:
        return 0
    else:
        return opp_history[-1]
    
rounds = 2000

opponents = {
    "Random": random_opponent,"Tit-for-Tat" : tit_for_tat
    }

payoff_matrix = {
    (0, 0): (3, 3),
    (0, 1): (0, 5),
    (1, 0): (5, 0),
    (1, 1): (0, 0)
}


for name, opponents in opponents.items():
    print(f"testing bot vs {name}")
    bot = AdaptivePrisonerBot()
    bot_score = 0
    opp_score = 0
    

    for round_number in range (1, rounds+1):
        opp_move = opponents(bot.bot_history)
        bot_move = bot.act()

        bot_payoff, opp_payoff = payoff_matrix[(bot_move, opp_move)]
        bot.learn(bot_move, opp_move, bot_payoff)

        bot_score += bot_payoff
        opp_score += opp_payoff

        print(f"Round {round_number}: Bot={'C' if bot_move==0 else 'D'} | Opponent={'C' if opp_move==0 else 'D'}")

    print(f"Final Score after {rounds} rounds: Bot={bot_score}, Opponent={opp_score}")
    coop_percentage = bot.bot_history.count(0) / len(bot.bot_history) * 100
    print(f"Bot cooperation rate vs {name}: {coop_percentage:.1f}%")

