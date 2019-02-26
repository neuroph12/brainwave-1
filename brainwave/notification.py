"""Templated notifications."""

class AccuracyAndLossTemplate:
    """Base template for notification that includes accuracy and loss metrics.

    Attributes:
        model (str): model name
        epoch (int): epoch
        train_accuracy (float): training accuracy for the epoch
        train_loss (float): training loss for the epoch
        val_accuracy: (float): validation accuracy for the epoch
        val_loss (float): validation loss for the epoch
    """
    def __init__(self, model, epoch, train_accuracy, train_loss, val_accuracy, val_loss):
        self.model = model
        self.epoch = epoch
        self.train_loss = train_loss 
        self.train_accuracy = train_accuracy 
        self.val_loss = val_loss 
        self.val_accuracy = val_accuracy 


class AccuracyAndLossEmail(AccuracyAndLossTemplate):
    """Templated email notification that includes accuracy and loss metrics."""
    @property
    def subject(self):
        """Returns (str) subject line of email."""
        return "{} @ epoch {}".format(self.model, self.epoch)

    @property
    def content(self):
        """Returns (str) content of email."""
        return (
            "train acc {:.4f}, loss {:.4f}; val acc {:.4f}, loss {:.4f}"
            .format(self.train_accuracy, self.train_loss, self.val_accuracy, self.val_loss)
        )


class AccuracyAndLossTextMessage(AccuracyAndLossEmail):
    """Templated text message notification that includes accuracy and loss metrics."""
    @property
    def content(self):
        """Returns (str) content of text message."""
        return super().subject + ": " + super().content

    @property
    def subject(self):
        raise NotImplementedError("Text messages do not have subject lines")
