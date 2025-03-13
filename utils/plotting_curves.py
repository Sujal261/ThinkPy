import matplotlib.pyplot as plt
def loss_curve(loss, title):
    plt.figure(figsize=(10,6))
    plt.plot(loss, label = 'Training Loss', color = 'blue')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()