from supermemo2 import first_review, review

# first review
# using quality=4 as an example, read below for what each value from 0 to 5 represents
# review date would default to datetime.utcnow() (UTC timezone) if not provided
first_review = first_review(4, "2024-06-22")
# first_review prints { "easiness": 2.36, "interval": 1, "repetitions": 1, "review_datetime": "2024-06-23 01:06:02"))

# second review
second_review = review(4, first_review["easiness"], first_review["interval"], first_review["repetitions"], first_review["review_datetime"])
# or just unpack the first review dictionary
second_review = review(4, **first_review)
# second_review prints similar to example above.