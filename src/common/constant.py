# coding:utf-8
from transformers.optimization import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_constant_schedule_with_warmup
)

amr_tokens = ["dummy", "multi-sentence", ':snt1', "<sep>", "<url>", ':wiki', ':arg0-of', ':arg1-of', ':arg2-of', ':arg3-of', ':arg4-of', ':arg5-of', ':arg5', ':path-of', ':source-of', ':age-of', ':part-of',  ':location-of', ':example-of', ':subevent-of', ':instrument-of', ':quant-of', ':medium-of', ':op1-of', ':destination-of', ':ord-of', ':purpose-of', ':duration-of', ':accompanier-of', ':time-of', 
                    ':age', ':duration', ':year', ':mod', ':manner-of',
                    ':snt7', ':year2', ':op5', ':subset-of', ':dayperiod', ':quant', ':season', ':subevent',
                    ':op9', ':accompanier', ':op6', ':li', ':direction-of', ':domain-of', ':op13', ':op11', ':op4',
                    ':condition', ':op16', ':condition-of', ':arg8', ':domain', ':time', ':weekday', ':arg2',
                    ':poss', ':beneficiary-of', ':prep-by', ':snt2', ':prep-in', ':snt8', ':concession-of',
                    ':topic-of', ':scale', ':snt6', ':arg3', ':prep-for', ':medium', ':op2', ':prep-on',
                    ':beneficiary', ':snt11', ':op7', ':prep-as', ':frequency-of', ':arg7', ':unit',
                    ':op1', ':path', ':value', ':degree-of', ':direction', ':poss-of', ':ord', ':month', ':op10',
                    ':quarter', ':op14', ':prep-under', ':snt3', ':prep-against', ':arg6', ':location',
                    ':destination', ':consist-of', ':purpose', ':degree', ':extent-of', ':extent',
                    ':op8', ':conj-as-if', ':prep-from', ':snt10', ':snt9',
                    ':topic', ':calendar', ':prep-at', ':polite',
                    ':example', ':prep-out-of', ':day', ':name-of', ':prep-amid', ':prep-into', ':concession',
                    ':part', ':arg9', ':arg0', ':op19', ':century', ':prep-among',
                    ':instrument', ':source', ':op17', ':prep-with', ':compared-to',
                    ':prep-in-addition-to', ':snt5', ':frequency',
                    ':timezone', ':op3', ':prep-toward', ':prep-on-behalf-of', ':prep-without',
                    ':name', ':op15', ':prep-along-with', ':arg4', ':mode', ':prep-to', ':decade',
                    ':op12', ':polarity', ':range', ':snt4', ':P',
                    ':manner', ':op18', ':op20', ':era', ':arg1', '<AMR>']


sent_tokens = []
max_turn = 32
for idx in range(max_turn):         # ma
    amr_tokens.append(f":speaker{idx+1}")
    #amr_tokens.append(f"U<{idx+1}>")
    amr_tokens.append(f"u<{idx+1}>")
    #sent_tokens.append(f"U<{idx+1}>")
    sent_tokens.append(f"u<{idx+1}>")

for idx in range(max_turn):         # ma
    #amr_tokens.append(f"Speaker{idx+1}")
    amr_tokens.append(f"speaker{idx+1}")
    #sent_tokens.append(f"Speaker{idx+1}")
    sent_tokens.append(f"speaker{idx+1}")

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule_with_warmup,
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"

rel2idx = {
    "none": 0,
    "ARG1": 1,
    "ARG2": 2,
    "mod": 3,
    "ARG0": 4,
    "age": 5,
    "ARG1-of": 6,
    "quant": 7,
    "location": 8,
    "ARG0-of": 9,
    "time": 10,
    "condition": 11,
    "topic": 12,
    "name": 13,
    "poss": 14,
    "duration": 15,
    "op1": 16,
    "domain": 17,
    "frequency": 18,
    "degree": 19,
    "consist-of": 20,
    "wiki": 21,
    "concession": 22,
    "mode": 23,
    "op2": 24,
    "medium": 25,
    "source": 26,
    "ARG3": 27,
    "beneficiary": 28,
    "purpose": 29,
    "accompanier": 30,
    "op3": 31,
    "ARG2-of": 32,
    "polarity": 33,
    "manner": 34,
    "time-of": 35,
    "part-of": 36,
    "direction": 37,
    "ARG4": 38,
    "location-of": 39,
    "op4": 40,
    "op5": 41,
    "prep-without": 42,
    "instrument": 43,
    "destination": 44,
    "prep-on": 45,
    "concession-of": 46,
    "op6": 47,
    "example": 48,
    "li": 49,
    "unk": 50
}