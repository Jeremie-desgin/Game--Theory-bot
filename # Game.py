# Game 
from LrBot import AdaptivePrisonerBot
import random

def playgame(opponent_move,player_move):
    player_payoff, opponent_payoff = payoff_matrix[(player_move, opponent_move)]
    return player_payoff ,opponent_payoff

Cooperate = 'C'
Defect = 'D'

payoff_matrix = {
    (Cooperate, Cooperate): (3, 3),
    (Cooperate, Defect): (0, 5),
    (Defect, Cooperate): (5, 0),
    (Defect, Defect): (0, 0)
}


def get_player_move():
    while True:
        try:
            pm = input(str("Choose your move (C = Cooperate, D = Defect): ")).strip().upper()
            if pm not in ("C","D"):
                print("Only C or D are allowed!")
                continue
            elif pm == "Q":
                return None
                
            return pm
            ##break
            
        except Exception:
            continue; 


def convert_to_bot_format(move):
    return 0 if move == 'C' else 1 

def convert_to_letter(move):
    return "C" if move == 0 else "D"


bot = AdaptivePrisonerBot() 
print("=== Prisoner's Dilemma vs Learning Bot ===")

round_number = 1
while True:
    print(f"\n--- Round {round_number} ---")

    # Player move
    player_move = get_player_move()
    if player_move == None:
        print("Goodbye!")
        break 
    player_num = convert_to_bot_format(player_move)
    bot_num = bot.act()

    bot_move = convert_to_letter(bot_num)

    player_payoff, bot_payoff = payoff_matrix[(player_move, bot_move)]

    bot.learn(bot_num,player_num , bot_payoff)

    print(f"You played: {player_move}")
    print(f"Bot played: {bot_move}")
    print(f"Your reward: {player_payoff}")
    print(f"Bot reward: {bot_payoff}")


    print("=== DEBUGGING ===")
    print("Bot history:", bot.bot_history)
    print("Opponent history:", bot.opp_history)


    round_number += 1