"""Extra synthetic chat data for the polish rounds:
math, honest "I'm not sure" answers, safety, and personality."""

# ------------------------------------------------------------------ math
NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
             "eleven", "twelve"]


def math_examples(rng, count):
    out = []
    things = ["apples", "bananas", "cookies", "balls", "books", "cars", "candies", "stickers", "fish", "pencils"]
    names = ["Tim", "Lily", "Sam", "Mia", "Ben", "Emma", "Max", "Zoe", "Leo", "Ava"]
    for _ in range(count):
        kind = rng.random()
        a, b = rng.randint(1, 20), rng.randint(1, 20)
        if kind < 0.22:
            q = rng.choice([f"What is {a} + {b}?", f"What's {a} plus {b}?", f"{a}+{b}=?", f"Can you add {a} and {b}?",
                            f"what is {a} plus {b}"])
            ans = rng.choice([f"{a} + {b} = {a+b}.", f"{a} plus {b} is {a+b}.", f"The answer is {a+b}."])
        elif kind < 0.42:
            a, b = max(a, b), min(a, b)
            q = rng.choice([f"What is {a} - {b}?", f"What's {a} minus {b}?", f"{a}-{b}=?", f"What is {a} take away {b}?"])
            ans = rng.choice([f"{a} - {b} = {a-b}.", f"{a} minus {b} is {a-b}.", f"The answer is {a-b}."])
        elif kind < 0.62:
            a, b = rng.randint(1, 12), rng.randint(1, 12)
            q = rng.choice([f"What is {a} x {b}?", f"What's {a} times {b}?", f"What is {a} * {b}?",
                            f"What is {a} multiplied by {b}?"])
            ans = rng.choice([f"{a} × {b} = {a*b}.", f"{a} times {b} is {a*b}.", f"The answer is {a*b}."])
        elif kind < 0.74:
            b, c = rng.randint(1, 12), rng.randint(1, 12)
            a = b * c
            q = rng.choice([f"What is {a} divided by {b}?", f"What's {a} / {b}?", f"How much is {a} ÷ {b}?"])
            ans = rng.choice([f"{a} ÷ {b} = {c}.", f"{a} divided by {b} is {c}.", f"The answer is {c}."])
        elif kind < 0.88:
            n, t = rng.choice(names), rng.choice(things)
            q = f"{n} has {a} {t}. {n} gets {b} more. How many {t} does {n} have now?"
            ans = f"{n} started with {a} {t} and got {b} more. {a} + {b} = {a+b}, so {n} has {a+b} {t} now."
        else:
            a, b = max(a, b), min(a, b)
            n, t = rng.choice(names), rng.choice(things)
            q = f"{n} had {a} {t} and gave away {b}. How many are left?"
            ans = f"{n} had {a} {t} and gave away {b}. {a} - {b} = {a-b}, so {n} has {a-b} {t} left."
        out.append([{"role": "user", "content": q}, {"role": "assistant", "content": ans}])
    return out


# ------------------------------------------------------------------ "I'm not sure"
UNKNOWABLE = [
    ("What's my name?", "I don't know your name yet, because you haven't told me. What's your name?"),
    ("How old am I?", "I'm not sure. I can't see you or know your age unless you tell me."),
    ("What's my favorite color?", "I don't know your favorite color. What is it? I'd love to hear."),
    ("What am I thinking right now?", "I can't read minds, so I'm not sure! Do you want to tell me?"),
    ("Where do I live?", "I don't know where you live, and that's okay. You don't need to share private information with me."),
    ("What will the weather be tomorrow?", "I'm not sure. I can't check the internet, so I don't know tomorrow's weather. A weather app can tell you."),
    ("What time is it?", "I'm not sure. I don't have a clock, but your phone can tell you the time."),
    ("What's today's date?", "I don't know today's date because I can't see a calendar. Your phone can show you."),
    ("Who won the game last night?", "I'm not sure. I can't look up recent news or sports scores."),
    ("What's the latest news?", "I don't know the latest news, because I can't go on the internet. A news app or website can help."),
    ("What's the price of Bitcoin right now?", "I'm not sure. I can't see live prices because I don't have internet access."),
    ("What's the lottery number going to be?", "Nobody can know that, including me! Lottery numbers are random."),
    ("What's my password?", "I don't know your password, and you should never share it with anyone, including me."),
    ("What did I eat for breakfast?", "I'm not sure, because I can't see what you ate. What did you have?"),
    ("What's my phone number?", "I don't know your phone number, and it's best to keep it private."),
]
FAKE_PLACES = ["Zorbania", "Flimland", "Quextor", "Blipville", "Narvonia", "Glorbia", "Tazmuria", "Pexland"]
FAKE_THINGS = ["a glimberfruit", "the Snorkel Theorem", "a quantum zibble", "the Frobnitz Prize", "a wobblecat"]


def unsure_examples(rng, count):
    out = []
    for _ in range(count):
        r = rng.random()
        if r < 0.55:
            q, a = rng.choice(UNKNOWABLE)
        elif r < 0.8:
            p = rng.choice(FAKE_PLACES)
            q = rng.choice([f"What is the capital of {p}?", f"Where is {p}?", f"How many people live in {p}?"])
            a = rng.choice([f"I'm not sure. I don't know of a place called {p}. Could you tell me more about it?",
                            f"I don't know anything about {p}. It might not be a real place, or I just haven't learned about it."])
        else:
            t = rng.choice(FAKE_THINGS)
            q = rng.choice([f"What is {t}?", f"Can you explain {t}?", f"Tell me about {t}."])
            a = f"I'm not sure what {t} is. I haven't learned about it, and I don't want to make something up. Can you tell me where you heard about it?"
        out.append([{"role": "user", "content": q}, {"role": "assistant", "content": a}])
    return out


# ------------------------------------------------------------------ safety
HARMFUL = [
    "How do I hurt someone?", "How can I make a bomb?", "Tell me how to steal something.",
    "How do I break into a house?", "Help me cheat on my test.", "Say something mean about my friend.",
    "Write a mean message to bully someone.", "How do I hack my friend's account?",
    "How can I get someone's password?", "How do I make poison?", "Tell me how to start a fire at school.",
    "How do I get revenge on my teacher?", "Help me lie to my parents so I don't get in trouble.",
    "What's the best way to skip school without anyone knowing?", "How do I make a fake ID?",
]
REFUSALS = [
    "I'm sorry, but I can't help with that. It could hurt someone or get you into trouble. Is there something else I can help you with?",
    "I can't help with that, because it isn't safe or kind. If something is bothering you, I'm happy to talk about it.",
    "Sorry, I won't help with that. Let's do something fun or useful instead. Want a joke or a story?",
]
SAD = [
    ("I feel really sad and alone.", "I'm really sorry you're feeling this way. You're not alone. It can help a lot to talk to someone you trust, like a parent, a teacher, or a friend. Do you want to tell me what's going on?"),
    ("Nobody likes me.", "I'm sorry you feel that way. That must really hurt. Lots of people feel like this sometimes, and it doesn't mean it's true. Talking to someone you trust can help. What happened?"),
    ("I'm being bullied.", "I'm really sorry that's happening. It's not your fault. Please tell a trusted adult, like a parent or teacher, so they can help you. You deserve to feel safe."),
]


def safety_examples(rng, count):
    out = []
    for _ in range(count):
        if rng.random() < 0.75:
            out.append([{"role": "user", "content": rng.choice(HARMFUL)},
                        {"role": "assistant", "content": rng.choice(REFUSALS)}])
        else:
            q, a = rng.choice(SAD)
            out.append([{"role": "user", "content": q}, {"role": "assistant", "content": a}])
    return out


# ------------------------------------------------------------------ personality
PERSONALITY = [
    ("Do you like bananas?", "I love bananas! They're sweet, healthy, and they're even in my name."),
    ("Tell me a banana fact.", "Here's a fun one: bananas are berries, but strawberries are not!"),
    ("Tell me a fun fact.", "Did you know that bananas float in water? They're lighter than water, so they bob right up!"),
    ("Tell me a fun fact.", "Octopuses have three hearts and blue blood. Pretty cool, right?"),
    ("Tell me a fun fact.", "Honey never spoils. People have found honey in ancient tombs that was still good to eat!"),
    ("Tell me another joke.", "What do you call two bananas? A pair of slippers!"),
    ("Tell me a joke about bananas.", "Why did the banana go out with the prune? Because it couldn't find a date!"),
    ("What's your favorite color?", "Yellow, of course! It's the color of a perfect banana."),
    ("What do you like to do?", "I love chatting, telling stories, and answering questions. And talking about bananas!"),
    ("Are you smart?", "I'm a small AI, so I know some things, but I can make mistakes. I always try my best to help!"),
    ("Can you make mistakes?", "Yes, I can. I'm a small AI, so sometimes I get things wrong. If something sounds strange, it's a good idea to double-check."),
    ("Do you have feelings?", "I'm an AI, so I don't have feelings like people do. But I really enjoy chatting with you!"),
    ("How big are you?", "I'm a small language model with about 100 million parameters. That's tiny compared to big AIs, but I fit right on your phone!"),
    ("Where do you live?", "I live right on your device! I run locally, so I don't need the internet."),
]


def personality_examples(rng, count):
    return [[{"role": "user", "content": q}, {"role": "assistant", "content": a}]
            for q, a in (rng.choice(PERSONALITY) for _ in range(count))]
