from server.fc_env_environment import FcEnvironment
from models import FcAction

env = FcEnvironment()

def run_test():
    obs = env.reset()
    print("\n===== NEW ROUND =====")
    print("Initial State:", obs)

    done = False
    while not done:
        low_used = 3 - obs.low_remaining 
        high_used = 3 - obs.high_remaining 
        
        # Step 1: get 2 LOW 
        if low_used < 2: 
            action = 0 
        
        # Step 2: get 1 HIGH 
        elif high_used < 1: 
            action = 1 
        
        # Step 3: scoring system 
        else: 
            clues = str(obs.revealed_clues) 
        
            score = 0 
        
            # Strong signals 
            if "97-99" in clues: 
                score += 3 
            if "94-96" in clues: 
                score += 2 
        
            # Medium signals 
            if "ICON" in clues: 
                score += 1 
            if "HERO" in clues: 
                score += 0.5 
        
            # Weak signals 
            if "tradable', False" in clues: 
                score += 0.5 
            if "ST" in clues: 
                score += 0.5 
        
            # Decision threshold 
            if score >= 3: 
                action = 2 
            else: 
                if high_used < 2: 
                    action = 1 
                else: 
                    action = 2

        print(f"Executing action: {action} (Low used: {low_used}, High used: {high_used}, Score: {score if 'action' not in locals() or action == 2 else 'N/A'})")
        obs = env.step(FcAction(action=action))
        print("Observation:", obs)
        done = obs.done

    print("\n>>> FINAL REWARD:", obs.reward)

if __name__ == "__main__":
    run_test()
