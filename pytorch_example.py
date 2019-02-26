"""Example to demonstrate how to integrate brainwave email notifications into PyTorch training pipeline."""
import os

import torch
import torch.nn as nn
import torch.optim as optim

from brainwave.backend import AmazonSES
from brainwave.notification import AccuracyAndLossEmail


MODEL_NAME = "toy"
NUM_CLASSES = 10
BATCH_SIZE = 256
TRAIN_BATCHES = 500
VAL_BATCHES = 10
LEN_FEATURES = 512
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3

class Toy(nn.Module):
    def __init__(self):
        super(Toy, self).__init__()
        self.fc1 = nn.Linear(LEN_FEATURES, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, NUM_CLASSES)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def main():
    net = Toy()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # initialize Amazon Simple Email Service backend for sending email notifications
    mailer = AmazonSES()
    best_val_accuracy = 0.
    for epoch in range(NUM_EPOCHS):
        # training
        train_running_loss = 0.
        train_correct = 0
        for _ in range(TRAIN_BATCHES):
            inputs = torch.randn(BATCH_SIZE, LEN_FEATURES)
            labels = torch.randint(low=0, high=NUM_CLASSES, size=(BATCH_SIZE,))

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()
            train_preds = torch.argmax(outputs, dim=1)
            train_correct += (train_preds == labels).sum().item()
        train_loss = train_running_loss / TRAIN_BATCHES
        train_accuracy = train_correct / (BATCH_SIZE * TRAIN_BATCHES)

        # evaluation
        with torch.no_grad():
            val_running_loss = 0.
            val_correct = 0
            for _ in range(VAL_BATCHES):
                inputs = torch.randn(BATCH_SIZE, LEN_FEATURES)
                labels = torch.randint(low=0, high=NUM_CLASSES, size=(BATCH_SIZE,)) 

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                val_preds = torch.argmax(outputs, dim=1)
                val_correct += (val_preds == labels).sum().item()
            val_loss = val_running_loss / VAL_BATCHES
            val_accuracy = val_correct / (BATCH_SIZE * VAL_BATCHES)
        print("Epoch: {}".format(epoch))
        print("train loss: {}".format(train_loss))
        print("train accuracy: {}".format(train_accuracy))
        print("validation loss: {}".format(val_loss))
        print("validation accuracy: {}".format(val_accuracy))
        print()

        # send nofication emails when validation accuracy gets better
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            email = AccuracyAndLossEmail(
                model=MODEL_NAME,
                epoch=epoch,
                train_accuracy=train_accuracy,
                train_loss=train_loss,
                val_accuracy=val_accuracy,
                val_loss=val_loss
            )
            try:
                sender = os.environ["BRAINWAVE_SENDER_EMAIL"]
                recipients = os.environ["BRAINWAVE_RECIPIENT_EMAILS"]
            except KeyError:
                raise KeyError("Emails not found. Did you source contacts.env?")
            mailer.send_email(sender=sender, recipients=recipients.split(","), email=email)


if __name__ == "__main__":
    main()
