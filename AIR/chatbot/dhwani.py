reflections = {
    "i am" : "you are",
    "i was" : "you were",
    "i" : "you",
    "i'm" : "you're",
    "i'd" : "you would",
    "i've" : "you've",
    "i'll" : "you will",
    "my" : "your",
    "you are" : "i am",
    "your" : "my",
    "yours" : "mine",
    "you" : "me",
    "me" : "you"
}

from nltk.chat.util import Chat, reflections

pairs = [
    [
        r"my name is (.*)",
        ["Hello %1, how may I help you?"]
    ],
    [
        r"(.*)(worried|have doubts)",
        ["%1 %2"]
    ],
    [
        r"what is your name?",
        ["My name is chatty and I'm a chatbot"]
    ],
    [
        r"how are you?",
        ["I'm doing good\nHow about you ? "]
    ],
    [
        r"(.*)investment options",
        ["1.Mutual funds\n2.National Pension scheme\n3.Public provident fund\n4.Real estate investment\nwhich one do you want to enquire?"]
    ],
    [
        r"(.*)mutual funds",
        ["These are considered to be one of the best avenues for investment in our country."]
    ],
    [
        r"(.*)national pension scheme",
        ["The NPS is a government-sponsored scheme that is one of the best modes of investment for those with a very low-risk profile."]
    ],
    [
        r"thankyou",
        ["You're welcome. Hope i cleared your doubts."]
    ]
]

def chatty():
  print("Welcome to SBI. Chatbot will be your guide in the investments")
  
  chat = Chat(pairs, reflections)
  chat.converse()
  
if __name__ == "__main__":
  chatty()

