from fractions import Fraction

AUTHORS = ["bach", "beethoven", "einaudi", "mozart"]

MIN_PS = 48  # minimum value for pitch space
MAX_PS = 86  # maximum value for pitch space
# Values for duration
ALL_DURATIONS = [0.0, 0.00390625, 0.02734375, 0.05859375, 0.0625,
                 Fraction(1, 12), Fraction(1, 10), 0.125, Fraction(1, 6),
                 0.19921875, 0.25, Fraction(1, 3), 0.375, 0.5, Fraction(2, 3),
                 0.75, 0.875, 1.0, 1.5, 1.75, 2.0, 3.0, 4.0, 6.0]

GENERATED_SONG_SIZE = 100
MAX_STEPS = 120
