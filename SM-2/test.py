from anki_sm_2 import Scheduler, Card, Rating, ReviewLog
from datetime import datetime, timezone

scheduler = Scheduler()

cards = [Card() for _ in range(9)]

# Rating.Fail (==0) forgot the card
# Rating.Pass (==1) remembered the card

review_logs : list[ReviewLog] = [None]*9 # pyright: ignore[reportAssignmentType]

print("Review 1")
ratings : list[Rating] = [Rating.Again] * 5 + [Rating.Good] * 5
for i in range(9):
    cards[i], review_logs[i] = scheduler.review_card(cards[i], ratings[i])

due_dates : list[datetime] = [card.due for card in cards] # pyright: ignore[reportAssignmentType]

# how much time between when the card is due and now
time_delta = [(due - datetime.now(timezone.utc)).total_seconds() for due in due_dates]  # pyright: ignore[reportOptionalOperand]

print(time_delta)

print("Review 2")

ratings : list[Rating] = [Rating.Again] * 3 + [Rating.Good] * 7
for i in range(9):
    cards[i], review_logs[i] = scheduler.review_card(cards[i], ratings[i])

due_dates : list[datetime] = [card.due for card in cards] # pyright: ignore[reportAssignmentType]

# how much time between when the card is due and now
time_delta = [(due - datetime.now(timezone.utc)).total_seconds() for due in due_dates]  # pyright: ignore[reportOptionalOperand]

print(time_delta)