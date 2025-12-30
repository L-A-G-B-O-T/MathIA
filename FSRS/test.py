from fsrs import Scheduler, Card, Rating, ReviewLog
import numpy as np
import heapq as pq

class FSRS_Train_Card:
    def __init__(self, x, y, card=Card()):
        self.x = x
        self.y = y
        self.card = card
    def __lt__(self, other):
        return self.card.due < other.card.due

scheduler = Scheduler()

# NOTE: all new cards are due immediately upon creation
cards = np.array([Card() for i in range(5)])
review_logs = np.array([None]*5)

# Rating.Again (==1) forgot the card
# Rating.Hard (==2) remembered the card with serious difficulty
# Rating.Good (==3) remembered the card after a hesitation
# Rating.Easy (==4) remembered the card easily

rating = np.ones(5)

cards[1], review_logs[1] = scheduler.review_card(cards[1], Rating.Good)

def review_datetime(rl):
    return rl.review_datetime

print(review_datetime(review_logs[1]))
# > Card rated 3 at 2024-11-30 17:46:58.856497+00:00

due = lambda card : card.due

# how much time between when the card is due and now
time_delta = due(cards[1]) - review_datetime(review_logs[1])

print(f"Card due in {time_delta} seconds")

cards[1], review_logs[1] = scheduler.review_card(cards[1], Rating.Good)

due = lambda card : card.due

# how much time between when the card is due and now
time_delta = due(cards[1]) - review_datetime(review_logs[1])

print(f"Card due in {time_delta} seconds")

# > Card due on 2024-11-30 18:42:36.070712+00:00
# > Card due in 599 seconds