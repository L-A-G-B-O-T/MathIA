from anki_sm_2 import Scheduler, Card, Rating

scheduler = Scheduler()

card = Card()

# Rating.Again (==1) forgot the card
# Rating.Hard (==2) remembered the card with serious difficulty
# Rating.Good (==3) remembered the card after a hesitation
# Rating.Easy (==4) remembered the card easily

rating = Rating.Good

card, review_log = scheduler.review_card(card, rating)

print(f"Card rated {review_log.rating} at {review_log.review_datetime}")
# > Card rated 3 at 2024-10-31 01:36:57.080934+00:00

from datetime import datetime, timezone

due = card.due

# how much time between when the card is due and now
time_delta = due - datetime.now(timezone.utc)

print(f"Card due: at {due}")
print(f"Card due in {time_delta.seconds / 60} minutes")
# > Card due: at 2024-10-31 01:46:57.080934+00:00
# > Card due in 9.983333333333333 minutes