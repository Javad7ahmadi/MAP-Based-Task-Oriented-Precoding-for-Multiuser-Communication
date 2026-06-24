print("Press 1 or 2 for selecting the trainer:")
print("1: MCR2")
print("2: Proposed")

choice = input("Enter your choice: ")

if choice == "1":
    MODE = "MCR2"
elif choice == "2":
    MODE = "OURLOSS"
else:
    print("Invalid choice. Defaulting to MCR2.")
    MODE = "MCR2"

BATCH_SIZE_TRAIN = 800
BATCH_SIZE_TEST = 64

epsilon_MCR2 = 0.9

K = 2
D_K = 12

WEIGHTS = "IMAGENET1K_V1"

chosen_classes = [53, 8, 99, 0, 85, 49, 66, 41, 15, 22]

num_classes = len(chosen_classes)

print(f"Selected MODE: {MODE}")