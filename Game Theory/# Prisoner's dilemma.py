# Prisoner's dilemma 
import random

Cooperate = 'C'
Defect = 'D'

payoff_matrix = {
    (Cooperate, Cooperate): (3, 3),
    (Cooperate, Defect): (0, 5),
    (Defect, Cooperate): (5, 0),
    (Defect, Defect): (0, 0)
}

for i in range(5):

    player_move = input("Choose your move (C = Cooperate, D = Defect): ").strip().upper()

    def play_round(player_move):
        opponent_move = random.choice([Cooperate, Defect])
        player_payoff, opponent_payoff = payoff_matrix[(player_move, opponent_move)]
        return opponent_move, player_payoff, opponent_payoff
    
    opponent_move, player_payoff, opponent_payoff = play_round(player_move)
    print(f"Round {i+1}: Opponent chose {opponent_move},  You chose {player_move}, "
          f"Your payoff = {player_payoff}, Opponent payoff = {opponent_payoff}")
    


