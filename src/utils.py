import matplotlib.pyplot as plt
import torch


# Visual (I generated this code with ai)

# =============================================
# VISUALIZATION
# =============================================


def visualize_loss(epochs, train_loss_values, test_loss_values):
    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the training and testing lines
    plt.plot(range(epochs), train_loss_values, label='Train Loss', color='blue', linewidth=2, marker='o')
    plt.plot(range(epochs), test_loss_values, label='Test Loss', color='red', linewidth=2, marker='x')

    # Add labels and styling
    plt.title('Transformer Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show the graph!
    plt.show()



def test_model(model, vocab):
    with torch.inference_mode():

        i_am = torch.tensor(
            [vocab["i"], vocab["am"]]
        )

        out = model(i_am)

        out = out.tolist()

        next_word_index = torch.argmax(torch.tensor(out[len(out)-1]))

        print("i am ", [key for key, value in vocab.items() if value == next_word_index][0])
