import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from app.core.config import settings
import logging
from fastapi import BackgroundTasks

logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self):
        try:
            self.smtp_server = settings.MAIL_SERVER
            self.smtp_port = settings.MAIL_PORT
            self.sender_email = settings.MAIL_FROM
            self.username = settings.MAIL_USERNAME  # Add username for Mailtrap
            self.password = settings.MAIL_PASSWORD
            
            # Debug logging
            logger.info(f"SMTP Server: {self.smtp_server}")
            logger.info(f"SMTP Port: {self.smtp_port}")
            logger.info(f"Username: {self.username}")
            logger.info(f"Sender Email: {self.sender_email}")
            logger.info("Password: {'set' if self.password else 'not set'}")
            
            self.enabled = all([
                self.smtp_server,
                self.smtp_port,
                self.username,
                self.password,
                self.sender_email
            ])
            
            if not self.enabled:
                logger.warning("Email service disabled due to missing configuration")
                logger.warning("Missing settings: " + ", ".join([
                    setting for setting, value in {
                        "MAIL_SERVER": self.smtp_server,
                        "MAIL_PORT": self.smtp_port,
                        "MAIL_USERNAME": self.username,
                        "MAIL_PASSWORD": self.password,
                        "MAIL_FROM": self.sender_email
                    }.items() if not value
                ]))
        except Exception as e:
            logger.warning(f"Email service initialization failed: {str(e)}")
            self.enabled = False

    def send_share_notification(
        self,
        recipient_email: str,
        document_title: str,
        shared_by_name: str
    ):
        if not self.enabled:
            logger.info(f"Email service disabled. Would have sent notification to {recipient_email}")
            return False

        try:
            message = MIMEMultipart("alternative")
            message["Subject"] = f"{shared_by_name} shared a document with you"
            message["From"] = self.sender_email
            message["To"] = recipient_email

            html = f"""
            <html>
              <body>
                <h2>Document Shared</h2>
                <p>Hello,</p>
                <p>{shared_by_name} has shared the document "{document_title}" with you.</p>
                <p>You can access it by logging into your account.</p>
                <br>
                <p>Best regards,<br>Your App Team</p>
              </body>
            </html>
            """

            message.attach(MIMEText(html, "html"))

            logger.info(f"Attempting to connect to SMTP server {self.smtp_server}:{self.smtp_port}")
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.set_debuglevel(1)  # Enable SMTP debug output
                logger.info("Starting TLS")
                server.starttls()
                logger.info(f"Logging in with username: {self.username}")
                server.login(self.username, self.password)  # Use username instead of sender_email
                logger.info("Sending message")
                server.send_message(message)
                logger.info("Message sent successfully")

            return True
        except Exception as e:
            logger.error(f"Failed to send share notification: {str(e)}", exc_info=True)
            return False