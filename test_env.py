from server.fc_env_environment import FcEnvironment
from models import FcAction

env = FcEnvironment()

def print_state(obs):
    print("\n--- STATE ---")
    print("Clues:", obs.revealed_clues)
    print("Tokens:", obs.tokens)
    print("Step:", obs.step_number)
    print("Done:", obs.done)
    print("Reward:", obs.reward)
    print("-------------\n")

while True:
    obs = env.reset()
    print("\n===== NEW ROUND =====")
    print_state(obs)

    done = False

    while not done:
        print("Actions:")
        print("0 -> Reveal LOW clue (cheap)")
        print("1 -> Reveal HIGH clue (expensive)")
        print("2 -> STOP and claim")
        print("3 -> SKIP round")

        action = int(input("Enter action: "))

        obs = env.step(FcAction(action=action))
        print_state(obs)

        done = obs.done

    print(">>> FINAL REWARD:", obs.reward)

    cont = input("Play again? (y/n): ")
    if cont.lower() != 'y':
        break