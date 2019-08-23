from grammar import *
from world import *

def main():
    v = Vocab.sample()
    g = Grammar(v)
    w = World()
    out = []
    complete = set()
    to_generate = 1000
    while len(complete) < to_generate:

        command = g.sample()
        if command.words() in complete:
            continue
        meaning = command.meaning()
        if not g.coherent(meaning):
            continue
        for j in range(100):
            situation = w.sample()
            demo = situation.demonstrate(meaning)
            if demo is not None:
                out.append((command, demo))
                complete.add(command.words())
                if (len(complete) + 1) % 100 == 0:
                    print("{:5d} / {:5d}".format(len(complete) + 1, to_generate))
                break

    splits = defaultdict(list)
    for command, demo in out:
        print(" ".join(command.words()))
        split = g.assign_split(command, demo)
        splits[split].append((command, demo))
    print()

    for split, data in splits.items():
        print(split, len(data))

if __name__ == "__main__":
    main()

# TODO path validator
# TODO write to file
