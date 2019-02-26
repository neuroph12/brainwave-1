# Brainwave
Brainwave is a notification system for neural network training. If you ever find yourself having to repeatedly check on model training progress on a (remote) PC, then this may be a useful tool. Brainwave sends you mobile notifications about the training progress, so you may stay informed while enjoying other things.

## Backends
Brainwave currently leverages the following service providers, and implements them as backends for sending notifications.
- Amazon Simple Notification Service (SNS): for sending text messages to phones
- Amazon Simple Email Service (SES): for sending emails
- SendGrid: for sending emails

There are free tiers for above services up to certain limits.

## Installation
- Verified on Python 3.6. Other 3.x might work as well.
- For Amazon Web Services (AWS) backends, need to have an AWS account, follow awscli and boto3 setup guides.
- For SendGrid backend, need to have a SendGrid account, and follow SendGrid Python setup guides.
- use brainwave/ as a standalone package

## Usage
1. Before sending notifications, need to specify contact information, such as phone number and e-mail addresses. See "example_contacts.env" for more information.
2. If not receiving e-mail notifications, check junk/spam folder for e-mails sent from service providers.
3. If using Amazon Simple Email Service, you need to first verify both the sender and recipient e-mails in the AWS SES console to prove that you own them and prevent unauthorized use.

## Examples
There are 2 examples provided on how to integrate Brainwave into training pipelines:
- tensorflow_example.py: example TensorFlow training pipeline with Brainwave text message notification integration
- pytorch_example.py: example PyTorch training pipeline with Brainwave email notification integration
