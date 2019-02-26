"""Backend services for sending notifications."""
import logging
import os

import boto3
from botocore.exceptions import ClientError
import sendgrid


logging.basicConfig(format=logging.BASIC_FORMAT)
logger = logging.getLogger(__name__)

class AmazonSNS:
    """Amazon Simple Notification Service for sending text messages.
    
    Attributes:
        service (boto3.Client): boto3 SNS client for interfacing with Amazon SNS
    """
    def __init__(self):
        self.service = boto3.client("sns")
    
    def send_text(self, to_number, message):
        """Send text message.

        Args:
            to_number (str): a phone number with a plus sign, followed by country code, area code
                and number, e.g. +14151234567, an example San Francisco, CA, USA number.
            message (obj, notification): an instance of templated text message notification object
        Logs:
            errors to console, but does not interrupt training
        """
        try:
            self.service.publish(PhoneNumber=to_number, Message=message.content)
        except ClientError as e:
            logger.exception(e)


class AmazonSES:
    """Amazon Simple Email Service for sending emails.
    
    Attributes:
        service (boto3.Client): boto3 SES client for interfacing with Amazon SES
    """
    def __init__(self):
        self.service = boto3.client("ses")
    
    def send_email(self, sender, recipients, email, charset="UTF-8"):
        """Send email.

        Args:
            sender (str): sender email address
            recipients (list of str): recipient email addresses
            email (obj, notification): an instance of templated email notification object
            charset (str): character set, optional, defaults to UTF-8
        Logs:
            errors to console, but does not interrupt training
        """
        try:
            self.service.send_email(
                Source=sender,
                Destination={"ToAddresses": recipients},
                Message={
                    "Subject": {"Data": email.subject, "Charset": charset},
                    "Body": {"Text": {"Data": email.content, "Charset": charset}}
                }
            )
        except ClientError as e:
            logger.exception(e)


class SendGrid:
    """SendGrid service for sending emails.
    
    Attributes:
        service (sendgrid client): SendGrid API client for interfacing with SendGrid
    """
    def __init__(self):
        self.service = sendgrid.SendGridAPIClient(apikey=os.environ.get('SENDGRID_API_KEY'))
    
    def send_email(self, sender, recipients, email):
        """Send email.

        Args:
            sender (str): sender email address
            recipients (list of str): recipient email addresses
            email (obj, notification): an instance of templated email notification object
        Logs:
            errors to console, but does not interrupt training
        """
        try:
            to = [{"email": i} for i in recipients]
            data = {
                "personalizations": [{"to": to, "subject": email.subject}],
                "from": {"email": sender},
                "content": [{"type": "text/plain", "value": email.content}]
            }
            self.service.client.mail.send.post(request_body=data)
        except Exception as e:
            logger.exception(e)
